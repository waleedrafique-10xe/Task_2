from __future__ import annotations

import copy
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from GenAIQuant.logger import logger
from GenAIQuant.utils.dataset_utils_vl import iter_vl_dataset
from GenAIQuant.utils.dtype import parse_dtype, preferred_dtype_for_device

from .dtype_utils import _coerce_model_dtype, _vision_dtype_overrides
from .quantizer.int_linear_fake import QuantLinear

if TYPE_CHECKING:
    from GenAIQuant.config import Config
    from GenAIQuant.interfaces import ModuleInterface


def _stack_batch(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {}
    concat_dim0 = {
        "pixel_values",
        "pixel_values_videos",
    }
    grid_keys = {"image_grid_thw", "video_grid_thw"}
    flat_concat = {"second_per_grid_ts"}
    for key in samples[0]:
        tensors = [sample.get(key) for sample in samples]
        tensors = [tensor for tensor in tensors if isinstance(tensor, torch.Tensor)]
        if not tensors or tensors[0] is None:
            continue
        if key in concat_dim0:
            prepared = [
                tensor if tensor.dim() > 0 else tensor.unsqueeze(0)
                for tensor in tensors
            ]
            batch[key] = torch.cat(prepared, dim=0)
            continue
        if key in grid_keys:
            prepared = []
            for tensor in tensors:
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(0)
                prepared.append(tensor)
            batch[key] = torch.cat(prepared, dim=0)
            continue
        if key in flat_concat:
            prepared = [tensor.reshape(-1) for tensor in tensors]
            batch[key] = torch.cat(prepared, dim=0)
            continue
        processed: list[torch.Tensor] = []
        for tensor in tensors:
            if tensor is None:
                continue
            if tensor.dim() > 0 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            processed.append(tensor)
        if not processed:
            continue
        if processed[0].dim() == 0:
            batch[key] = torch.stack(processed)
            continue
        max_dims = [
            max(t.shape[dim] for t in processed) for dim in range(processed[0].dim())
        ]
        target_shape = (len(processed), *max_dims)
        stacked = processed[0].new_zeros(target_shape)
        for idx, tensor in enumerate(processed):
            dest = stacked[idx]
            slices = tuple(slice(0, size) for size in tensor.shape)
            dest[slices] = tensor
        batch[key] = stacked
    return batch


def _resolve_vision_token_ids(processor, configured_ids: list[int] | None) -> list[int]:
    if configured_ids:
        return sorted({int(token_id) for token_id in configured_ids})

    tokenizer = getattr(processor, "tokenizer", None)
    candidate_attrs = (
        "image_token_id",
        "vision_token_id",
        "img_token_id",
        "default_image_token_id",
    )
    discovered: set[int] = set()
    for attr in candidate_attrs:
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int):
            discovered.add(int(value))
    special_tokens = getattr(tokenizer, "additional_special_tokens_ids", None)
    if isinstance(special_tokens, list):
        for token in special_tokens:
            if not isinstance(token, int) or token < 0:
                continue
            token_str = tokenizer.convert_ids_to_tokens(token)
            if isinstance(token_str, str) and "image" in token_str.lower():
                discovered.add(int(token))
    return sorted(discovered)


def _mbq_logits_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    vision_token_ids: Sequence[int],
    tau: float,
    kd_weight: float,
    vision_weight: float,
    language_weight: float,
    reweight: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tau = max(float(tau), 1e-6)
    device = student_logits.device

    teacher_logits = teacher_logits.to(device=device, dtype=student_logits.dtype)
    student_logits = student_logits.to(device=device, dtype=student_logits.dtype)
    input_ids = input_ids.to(device=device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device).float()

    t_logp = F.log_softmax(teacher_logits / tau, dim=-1)
    s_logp = F.log_softmax(student_logits / tau, dim=-1)
    kld = F.kl_div(s_logp, t_logp.exp(), reduction="none").sum(dim=-1)

    if vision_token_ids:
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in vision_token_ids:
            mask = mask | (input_ids == int(token_id))
        vision_mask = mask.float()
    else:
        vision_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    language_mask = torch.ones_like(vision_mask, dtype=torch.float32) - vision_mask

    if attention_mask is not None:
        vision_mask = vision_mask * attention_mask
        language_mask = language_mask * attention_mask

    vision_sum = vision_mask.sum().clamp(min=1.0)
    language_sum = language_mask.sum().clamp(min=1.0)

    vision_loss = (kld * vision_mask).sum() / vision_sum
    language_loss = (kld * language_mask).sum() / language_sum

    if reweight:
        total = vision_sum + language_sum
        vision_scale = total / (2.0 * vision_sum)
        language_scale = total / (2.0 * language_sum)
    else:
        vision_scale = language_scale = 1.0

    total_loss = (
        vision_weight * vision_scale * vision_loss
        + language_weight * language_scale * language_loss
    )
    total_loss = kd_weight * (tau * tau) * total_loss
    return total_loss, vision_loss.detach(), language_loss.detach()


def _quantize_vision_layers(
    model: nn.Module,
    base_prefix: str,
    wbits: int,
    group_size: int,
    overrides: dict[str, int],
) -> list[tuple[str, QuantLinear]]:
    quant_modules: list[tuple[str, QuantLinear]] = []

    def replace(module: nn.Module, prefix: str):
        for name, child in list(module.named_children()):
            child_prefix = f"{prefix}.{name}" if prefix else name
            if isinstance(child, QuantLinear):
                quant_modules.append((child_prefix, child))
            elif isinstance(child, nn.Linear):
                bitwidth = overrides.get(child_prefix, wbits)
                quant_layer = QuantLinear(child, int(bitwidth), group_size)
                quant_layer.set_quant_state(True)
                setattr(module, name, quant_layer)
                quant_modules.append((child_prefix, quant_layer))
            else:
                replace(child, child_prefix)

    replace(model, base_prefix)
    return quant_modules


def _collect_quant_linear_pairs(
    model: nn.Module,
    *,
    include_prefixes: Sequence[str] | None = None,
    exclude_prefixes: Sequence[str] | None = None,
) -> list[tuple[str, QuantLinear]]:
    pairs: list[tuple[str, QuantLinear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, QuantLinear):
            continue
        if include_prefixes and not any(
            name.startswith(prefix) for prefix in include_prefixes
        ):
            continue
        if exclude_prefixes and any(
            name.startswith(prefix) for prefix in exclude_prefixes
        ):
            continue
        pairs.append((name, module))
    return pairs


def _freeze_for_scale_training(
    model: nn.Module, quant_modules: list[QuantLinear]
) -> list[nn.Parameter]:
    for param in model.parameters():
        param.requires_grad = False
    trainable: list[nn.Parameter] = []
    seen: set[int] = set()
    for quant_module in quant_modules:
        if hasattr(quant_module, "weight") and isinstance(
            quant_module.weight, nn.Parameter
        ):
            quant_module.weight.requires_grad = False
        quantizer = (
            quant_module.weight_quantizer
            if hasattr(quant_module, "weight_quantizer")
            else None
        )
        if quantizer is None:
            continue
        scale = quantizer.scale if hasattr(quantizer, "scale") else None
        zero_point = quantizer.zero_point if hasattr(quantizer, "zero_point") else None
        for param in (scale, zero_point):
            if param is None:
                continue
            param.requires_grad = True
            ident = id(param)
            if ident in seen:
                continue
            seen.add(ident)
            trainable.append(param)
    return trainable


def _sanitize_quant_params(modules: Sequence[QuantLinear]) -> None:
    """Clamp and denan quantizer parameters to keep training stable."""

    for module in modules:
        quantizer = getattr(module, "weight_quantizer", None)
        if quantizer is None:
            continue
        scale = getattr(quantizer, "scale", None)
        if isinstance(scale, torch.Tensor):
            scale_data = scale.data
            scale_data.nan_to_num_(nan=1e-3, posinf=1e3, neginf=1e-3)
            scale_data.clamp_(1e-6, 1e2)
        zero_point = getattr(quantizer, "zero_point", None)
        if isinstance(zero_point, torch.Tensor):
            zp_data = zero_point.data
            qmin = float(getattr(quantizer, "qmin", 0))
            qmax = float(getattr(quantizer, "qmax", 0))
            zp_data.nan_to_num_(nan=0.0, posinf=qmax, neginf=qmin)
            zp_data.clamp_(qmin, qmax)


def _resolve_device(
    value: torch.device | str | None,
    fallback: torch.device,
) -> torch.device:
    if value is None:
        return fallback
    if isinstance(value, torch.device):
        return value
    return torch.device(value)


def _move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    dtype_overrides: dict[str, torch.dtype] | None = None,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if (
                dtype_overrides is not None
                and key in dtype_overrides
                and value.dtype.is_floating_point
            ):
                result[key] = value.to(device=device, dtype=dtype_overrides[key])
            else:
                result[key] = value.to(device=device)
        else:
            result[key] = value
    result.setdefault("use_cache", False)
    return result


def _forward_logits(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    inputs = {
        "input_ids": batch.get("input_ids"),
        "attention_mask": batch.get("attention_mask"),
        "pixel_values": batch.get("pixel_values"),
        "pixel_values_videos": batch.get("pixel_values_videos"),
        "image_grid_thw": batch.get("image_grid_thw"),
        "use_cache": batch.get("use_cache", False),
    }
    inputs = {k: v for k, v in inputs.items() if v is not None}
    ref_param = next(model.parameters(), None)
    device_type = ref_param.device.type if ref_param is not None else "cpu"
    target_dtype = ref_param.dtype if ref_param is not None else None
    if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
        vision_module = model.visual if hasattr(model, "visual") else None
        vision_param = (
            next(vision_module.parameters(), None)
            if vision_module is not None
            else None
        )
        vision_dtype = vision_param.dtype if vision_param is not None else target_dtype
        if vision_dtype is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=vision_dtype)
    if "pixel_values_videos" in inputs and isinstance(
        inputs["pixel_values_videos"], torch.Tensor
    ):
        vision_module = model.visual if hasattr(model, "visual") else None
        vision_param = (
            next(vision_module.parameters(), None)
            if vision_module is not None
            else None
        )
        vision_dtype = vision_param.dtype if vision_param is not None else target_dtype
        if vision_dtype is not None:
            inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(
                dtype=vision_dtype
            )
    autocast_ctx = nullcontext()
    if device_type in {"cuda", "xpu"} and target_dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=target_dtype)
    with autocast_ctx:
        outputs = model(**inputs)
    logits = getattr(outputs, "logits", None)
    if logits is None:
        logits = outputs[0]
    return logits.float()


def _gather_quant_state(
    quant_modules: list[tuple[str, QuantLinear]],
) -> dict[str, dict[str, torch.Tensor]]:
    state: dict[str, dict[str, torch.Tensor]] = {}
    for name, module in quant_modules:
        quantizer = getattr(module, "weight_quantizer", None)
        if quantizer is None:
            continue
        state[name] = {
            "scale": quantizer.scale.detach().cpu(),
            "zero_point": quantizer.zero_point.detach().cpu(),
        }
    return state


def _load_quant_state(
    quant_modules: list[tuple[str, QuantLinear]],
    state: dict[str, dict[str, torch.Tensor]],
):
    for name, module in quant_modules:
        payload = state.get(name)
        if payload is None:
            continue
        quantizer = getattr(module, "weight_quantizer", None)
        if quantizer is None:
            continue
        scale_param = quantizer.scale if hasattr(quantizer, "scale") else None
        zero_param = quantizer.zero_point if hasattr(quantizer, "zero_point") else None
        if "scale" in payload and scale_param is not None:
            scale_param.data.copy_(payload["scale"].to(scale_param.device))
        if "zero_point" in payload and zero_param is not None:
            zero_param.data.copy_(payload["zero_point"].to(zero_param.device))


def main_vision(interface: ModuleInterface, processor, config: Config):
    prepared_model = interface.model
    qat_cfg = config.quantization.qat
    mbq_cfg = getattr(qat_cfg, "mbq", None)
    assert mbq_cfg is not None and mbq_cfg.enabled, "MBQ is not enabled."

    if processor is None:
        raise ValueError("Processor is required for multimodal MBQ.")

    if not getattr(mbq_cfg, "kd_enabled", True):
        raise ValueError("MBQ: kd_enabled must be True for modality-balanced KD.")

    dataset_name = getattr(mbq_cfg, "dataset_name", None)
    dataset_label = dataset_name or "<unspecified>"
    mbq_train_size = getattr(mbq_cfg, "train_size", None) or qat_cfg.train_size
    mbq_val_size = getattr(mbq_cfg, "val_size", None) or qat_cfg.val_size
    logger.info(
        "MBQ: using dataset '%s' (train_size=%d, val_size=%d, batch_size=%d)",
        dataset_label,
        mbq_train_size,
        mbq_val_size,
        mbq_cfg.batch_size,
    )

    device = torch.device(config.device)
    teacher_device = _resolve_device(getattr(mbq_cfg, "teacher_device", None), device)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    core_model = prepared_model.model.to("cpu")
    target_dtype = qat_cfg.target_dtype
    if not isinstance(target_dtype, torch.dtype):
        target_dtype = parse_dtype(target_dtype)
    if target_dtype is None:
        target_dtype = preferred_dtype_for_device(device)

    teacher = copy.deepcopy(core_model)
    teacher, teacher_dtype = _coerce_model_dtype(teacher, teacher_device, target_dtype)
    teacher.eval()

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    student, student_dtype = _coerce_model_dtype(core_model, device, target_dtype)
    student.eval()

    vision_module = prepared_model.get_vision_module(student)
    if vision_module is None:
        raise AttributeError(
            "Unable to resolve vision module from meta; ensure metas define vision_model_name or vision_layers."
        )

    overrides = getattr(mbq_cfg, "vision_bitwidth_overrides", {})
    base_prefix = prepared_model.vision_model_prefix
    if base_prefix is None:
        raise AttributeError("Vision model prefix could not be determined from meta.")

    quant_pairs = _quantize_vision_layers(
        model=vision_module,
        base_prefix=base_prefix,
        wbits=mbq_cfg.wbits,
        group_size=mbq_cfg.group_size,
        overrides=overrides,
    )
    if not quant_pairs:
        logger.warning("MBQ: found no vision linear modules to quantize.")
        return prepared_model
    quant_pair_map: dict[str, QuantLinear] = {
        name: module for name, module in quant_pairs
    }

    nonvision_pairs = _collect_quant_linear_pairs(
        student,
        exclude_prefixes=(base_prefix,),
    )
    added = 0
    for name, module in nonvision_pairs:
        if name in quant_pair_map:
            continue
        quant_pair_map[name] = module
        added += 1
    if added > 0:
        logger.info(
            "MBQ: added %d additional QuantLinear modules outside vision prefix '%s'.",
            added,
            base_prefix,
        )
    else:
        logger.info(
            "MBQ: no extra QuantLinear modules found outside vision prefix '%s'.",
            base_prefix,
        )

    quant_pairs = list(quant_pair_map.items())
    quant_modules = list(quant_pair_map.values())
    trainable_params = _freeze_for_scale_training(student, quant_modules)
    logger.info("MBQ: trainable quant modules=%d", len(quant_modules))

    if not trainable_params or mbq_cfg.epochs <= 0:
        logger.info("MBQ: nothing to train for vision tower.")
        return prepared_model

    train_iter, val_iter = iter_vl_dataset(
        processor=processor,
        dataset_name=mbq_cfg.dataset_name,
        train_size=mbq_train_size,
        val_size=mbq_val_size,
        seed=getattr(qat_cfg, "seed", 0),
        train_split="train",
        val_split="validation",
        max_length=qat_cfg.training_seqlen,
    )

    batch_size = max(1, mbq_cfg.batch_size)
    
    val_batches: list[dict[str, torch.Tensor]] = []
    val_buffer: list[dict[str, torch.Tensor]] = []
    for sample, _ in val_iter:
        val_buffer.append(sample)
        if len(val_buffer) == batch_size:
            val_batches.append(_stack_batch(val_buffer))
            val_buffer.clear()
    if val_buffer:
        val_batches.append(_stack_batch(val_buffer))

    train_batches_cache: list[dict[str, torch.Tensor]] = []

    def _iter_train_batches(epoch: int) -> Iterable[dict[str, torch.Tensor]]:
        if epoch == 1:
            buffer: list[dict[str, torch.Tensor]] = []
            for sample, _ in train_iter:
                buffer.append(sample)
                if len(buffer) == batch_size:
                    batch = _stack_batch(buffer)
                    train_batches_cache.append(batch)
                    yield batch
                    buffer = []
            if buffer:
                batch = _stack_batch(buffer)
                train_batches_cache.append(batch)
                yield batch
        else:
            for batch in train_batches_cache:
                yield batch

    if mbq_train_size <= 0:
        logger.warning("MBQ: non-positive training size; skipping vision QAT.")
        return prepared_model

    teacher_vision_module = (
        prepared_model.get_vision_module(teacher) if teacher is not None else None
    )
    teacher_dtype_overrides = _vision_dtype_overrides(teacher_vision_module)
    student_dtype_overrides = _vision_dtype_overrides(vision_module)

    vision_token_ids = _resolve_vision_token_ids(
        processor, getattr(mbq_cfg, "vision_token_ids", None)
    )
    if not vision_token_ids:
        logger.warning(
            "MBQ: no vision token ids were provided or inferred; treating all tokens as language."
        )

    base_lr = qat_cfg.quant_lr if qat_cfg.quant_lr > 0 else 1e-4
    lr = min(base_lr, 1e-4)
    optimizer = AdamW(
        trainable_params, lr=lr, weight_decay=qat_cfg.wd, betas=(0.9, 0.98), eps=1e-8
    )

    expected_steps_per_epoch = max(
        1, math.ceil(float(mbq_train_size) / float(batch_size))
    )
    total_steps = expected_steps_per_epoch * mbq_cfg.epochs
    warmup_steps = min(50, total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_steps, 1),
    )

    best_loss = float("inf")
    best_state: dict[str, dict[str, torch.Tensor]] | None = None

    for epoch in range(1, mbq_cfg.epochs + 1):
        student.train()
        running_total = 0.0
        running_vision = 0.0
        running_language = 0.0
        steps_in_epoch = 0
        for step, batch in enumerate(_iter_train_batches(epoch), start=1):
            steps_in_epoch = step
            teacher_inputs = _move_batch_to_device(
                batch,
                teacher_device,
                dtype_overrides=teacher_dtype_overrides,
            )
            student_inputs = _move_batch_to_device(
                batch,
                device,
                dtype_overrides=student_dtype_overrides,
            )
            with torch.no_grad():
                teacher_logits = _forward_logits(teacher, teacher_inputs)
            student_logits = _forward_logits(student, student_inputs)

            total_loss, vision_loss, language_loss = _mbq_logits_loss(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                vision_token_ids=vision_token_ids,
                tau=mbq_cfg.tau,
                kd_weight=mbq_cfg.kd_weight,
                vision_weight=mbq_cfg.vision_weight,
                language_weight=mbq_cfg.language_weight,
                reweight=bool(mbq_cfg.reweight),
            )

            if not torch.isfinite(total_loss):
                logger.warning(
                    "MBQ: non-finite loss at epoch %d step %d; skipping batch.",
                    epoch,
                    step,
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            max_norm = float(qat_cfg.clip_grad)
            if max_norm > 0:
                clip_grad_norm_(trainable_params, max_norm)
            optimizer.step()
            _sanitize_quant_params(quant_modules)
            scheduler.step()

            running_total += total_loss.detach().item()
            running_vision += vision_loss.item()
            running_language += language_loss.item()

            if step % max(1, expected_steps_per_epoch // 10) == 0:
                avg_total = running_total / max(step, 1)
                avg_vision = running_vision / max(step, 1)
                avg_language = running_language / max(step, 1)
                lr_val = scheduler.get_last_lr()[0]
                logger.info(
                    "[MBQ] epoch %d step %d/%d total %.4f (vision %.4f, language %.4f) lr %.2e",
                    epoch,
                    step,
                    expected_steps_per_epoch,
                    avg_total,
                    avg_vision,
                    avg_language,
                    lr_val,
                )

        if epoch == 1 and steps_in_epoch == 0:
            logger.warning("MBQ: no training batches available; skipping vision QAT.")
            return prepared_model

        if epoch == 1:
            logger.info(
                "MBQ: prepared %d training batches from dataset (requested %d samples).",
                len(train_batches_cache),
                mbq_train_size,
            )

        student.eval()
        eval_totals: list[float] = []
        eval_vision_losses: list[float] = []
        eval_language_losses: list[float] = []

        for idx, batch in enumerate(val_batches):
            if idx >= 5:
                break
            teacher_inputs = _move_batch_to_device(
                batch,
                teacher_device,
                dtype_overrides=teacher_dtype_overrides,
            )
            student_inputs = _move_batch_to_device(
                batch,
                device,
                dtype_overrides=student_dtype_overrides,
            )
            with torch.no_grad():
                teacher_logits = _forward_logits(teacher, teacher_inputs)
                student_logits = _forward_logits(student, student_inputs)
                eval_total, eval_vision, eval_language = _mbq_logits_loss(
                    teacher_logits=teacher_logits,
                    student_logits=student_logits,
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    vision_token_ids=vision_token_ids,
                    tau=mbq_cfg.tau,
                    kd_weight=mbq_cfg.kd_weight,
                    vision_weight=mbq_cfg.vision_weight,
                    language_weight=mbq_cfg.language_weight,
                    reweight=bool(mbq_cfg.reweight),
                )
            eval_totals.append(eval_total.item())
            eval_vision_losses.append(eval_vision.item())
            eval_language_losses.append(eval_language.item())

        mean_total = sum(eval_totals) / max(len(eval_totals), 1)
        mean_vision = sum(eval_vision_losses) / max(len(eval_vision_losses), 1)
        mean_language = sum(eval_language_losses) / max(len(eval_language_losses), 1)
        logger.info(
            "[MBQ] epoch %d eval total %.4f (vision %.4f, language %.4f)",
            epoch,
            mean_total,
            mean_vision,
            mean_language,
        )

        if mean_total < best_loss:
            best_loss = mean_total
            best_state = _gather_quant_state(quant_pairs)
            payload = {
                "quant_modules": best_state,
                "wbits": mbq_cfg.wbits,
                "group_size": mbq_cfg.group_size,
                "mbq_config": {
                    "vision_weight": mbq_cfg.vision_weight,
                    "language_weight": mbq_cfg.language_weight,
                    "reweight": bool(mbq_cfg.reweight),
                    "tau": mbq_cfg.tau,
                    "kd_weight": mbq_cfg.kd_weight,
                    "vision_token_ids": vision_token_ids,
                },
            }
            torch.save(payload, output_dir / "mbq_best_scales.pt")

    final_state = _gather_quant_state(quant_pairs)
    torch.save(
        {
            "quant_modules": final_state,
            "wbits": mbq_cfg.wbits,
            "group_size": mbq_cfg.group_size,
            "mbq_config": {
                "vision_weight": mbq_cfg.vision_weight,
                "language_weight": mbq_cfg.language_weight,
                "reweight": bool(mbq_cfg.reweight),
                "tau": mbq_cfg.tau,
                "kd_weight": mbq_cfg.kd_weight,
                "vision_token_ids": vision_token_ids,
            },
        },
        output_dir / "mbq_final_scales.pt",
    )

    if best_state is not None:
        _load_quant_state(quant_pairs, best_state)

    teacher.cpu()
    if teacher_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prepared_model
