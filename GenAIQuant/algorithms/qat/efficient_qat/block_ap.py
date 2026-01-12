import copy
import gc
import math
import os
import shutil
import time
from collections.abc import Mapping

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .datautils_block import BlockTrainDataset
from .quantizer import int_linear_real
from .quantizer.int_linear_fake import QuantLinear
from .utils import (
    NativeScalerWithGradNormCount,
    get_named_linears,
    quant_inplace,
    quant_parameters,
    set_op_by_name,
    set_quant_parameters,
    set_quant_state,
    set_weight_parameters,
    trainable_parameters,
    trainable_parameters_num,
    weight_parameters,
)


_DEFAULT_PAD_KEYS = {"input_ids": lambda pad_id: pad_id}
_NO_PAD_KEYS = {
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "second_per_grid_ts",
}


def _collate_calibration_batch(samples, device, pad_token_id):
    first = samples[0]
    if isinstance(first, torch.Tensor):
        batch = torch.cat(samples, dim=0)
        return batch.to(device)

    seq_len = None
    if any("input_ids" in sample for sample in samples):
        seq_len = max(
            sample["input_ids"].shape[-1] for sample in samples if "input_ids" in sample
        )

    collated: dict[str, torch.Tensor] = {}
    keys = set().union(*(sample.keys() for sample in samples))
    for key in keys:
        tensors = [sample[key] for sample in samples if key in sample]
        if not tensors:
            continue
        needs_pad = (
            seq_len is not None
            and key not in _NO_PAD_KEYS
            and any(
                tensor.ndim >= 1 and tensor.shape[-1] != seq_len for tensor in tensors
            )
        )
        if needs_pad and seq_len is not None:
            pad_value = 0
            if key in _DEFAULT_PAD_KEYS:
                pad_value = _DEFAULT_PAD_KEYS[key](pad_token_id)
            tensors = [
                torch.nn.functional.pad(
                    tensor,
                    (0, seq_len - tensor.shape[-1]),
                    value=pad_value,
                )
                if tensor.shape[-1] < seq_len
                else tensor[..., :seq_len]
                for tensor in tensors
            ]
        collated[key] = torch.cat(tensors, dim=0).to(device)
    return collated


def _ensure_torch_dtype(value, fallback=torch.float16):
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        attr = value.lower()
        if hasattr(torch, attr):
            candidate = torch.__dict__[attr]
        else:
            candidate = None
        if isinstance(candidate, torch.dtype):
            return candidate
    return fallback


def _invoke_model_forward(model, batch_inputs):
    if isinstance(batch_inputs, torch.Tensor):
        return model(batch_inputs)
    if isinstance(batch_inputs, Mapping):
        return model(**batch_inputs)
    raise TypeError(f"Unsupported calibration batch type: {type(batch_inputs)}")


def _repeat_first_dim_if_needed(tensor, target_batch):
    if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
        return tensor
    if tensor.shape[0] == target_batch or target_batch <= 1:
        return tensor
    if tensor.shape[0] == 1:
        repeat_dims = [target_batch] + [1] * (tensor.ndim - 1)
        return tensor.repeat(*repeat_dims)
    return tensor


def _repeat_position_embeddings_tuple(position_embeddings, target_batch, device):
    if position_embeddings is None:
        return None
    repeated = []
    for tensor in position_embeddings:
        if tensor is None:
            repeated.append(None)
            continue
        rep = _repeat_first_dim_if_needed(tensor, target_batch)
        repeated.append(rep.to(device))
    return tuple(repeated)


def update_dataset(
    layer,
    dataset,
    dev,
    attention_mask,
    position_ids,
    position_embeddings=None,
    position_embeddings_global=None,
    position_embeddings_local=None,
):
    device_obj = torch.device(dev)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device_obj.type):
            for index, inps in enumerate(dataset):
                inps = inps.to(device_obj)
                if inps.ndim == 2:
                    inps = inps.unsqueeze(0)
                batch_size = inps.shape[0]
                seq_len = inps.shape[1]
                cache_position = torch.arange(seq_len, device=device_obj)
                attn_mask = _repeat_first_dim_if_needed(attention_mask, batch_size)
                if isinstance(attn_mask, torch.Tensor):
                    attn_mask = attn_mask.to(device_obj).float()
                pos_ids = _repeat_first_dim_if_needed(position_ids, batch_size)
                if isinstance(pos_ids, torch.Tensor):
                    pos_ids = pos_ids.to(device_obj)
                positional_embeddings = _repeat_position_embeddings_tuple(
                    position_embeddings, batch_size, device_obj
                )
                pos_emb_global = _repeat_first_dim_if_needed(
                    position_embeddings_global, batch_size
                )
                if isinstance(pos_emb_global, torch.Tensor):
                    pos_emb_global = pos_emb_global.to(device_obj)
                pos_emb_local = _repeat_first_dim_if_needed(
                    position_embeddings_local, batch_size
                )
                if isinstance(pos_emb_local, torch.Tensor):
                    pos_emb_local = pos_emb_local.to(device_obj)

                new_data = layer(
                    inps,
                    attention_mask=attn_mask,
                    position_ids=pos_ids,
                    position_embeddings=positional_embeddings,
                    position_embeddings_global=pos_emb_global,
                    position_embeddings_local=pos_emb_local,
                    cache_position=cache_position,
                    use_cache=False,
                )[0].to("cpu")
                dataset.update_data(index, new_data)


def block_ap(
    model,
    config,
    pad_token_id,
    wbits_config,
    trainloader,
    valloader,
    device,
    logger=None,
):
    logger.info("Starting ...")
    if config.off_load_to_disk:
        logger.info(
            "offload the training dataset to disk, saving CPU memory, but may slowdown the training due to additional I/O..."
        )

    resolved_device = (
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    device_obj = torch.device(resolved_device)
    if hasattr(config, "target_dtype"):
        config_target_dtype = config.target_dtype
    else:
        config_target_dtype = torch.float16
    dtype = _ensure_torch_dtype(config_target_dtype)
    use_cache = model.model_config.use_cache
    model.model.config.use_cache = False

    layers = model.layers
    model.embedding = model.embedding.to(device=device_obj, dtype=dtype)
    model.model_norm = model.model_norm.to(device=device_obj, dtype=dtype)
    if hasattr(model, "rope"):
        model.rope = model.rope.to(device_obj)
    layers[0] = layers[0].to(device=device_obj, dtype=dtype)

    if hasattr(config, "train_size_in_memory"):
        train_mem_limit = config.train_size_in_memory
    else:
        train_mem_limit = None
    use_ring_buffer = (
        train_mem_limit is not None
        and train_mem_limit > 0
        and train_mem_limit < config.train_size
    )
    if use_ring_buffer:
        if config.off_load_to_disk and logger:
            logger.info("Ignoring off_load_to_disk because ring-buffer mode streams data on the fly.")
        logger.info(
            "Ring-buffer calibration enabled: keeping %d samples in memory (total=%d).",
            train_mem_limit,
            config.train_size,
        )
        _block_ap_ring_buffer(
            model=model,
            config=config,
            pad_token_id=pad_token_id,
            wbits_config=wbits_config,
            trainloader=trainloader,
            valloader=valloader,
            device=device,
            logger=logger,
            device_obj=device_obj,
            dtype=dtype,
        )
        model.model_config.use_cache = use_cache
        return model

    flag = time.time()
    if config.off_load_to_disk:
        fp_train_cache_path = f"{config.cache_dir}/{flag}/block_training_fp_train"
        fp_val_cache_path = f"{config.cache_dir}/{flag}/block_training_fp_val"
        quant_train_cache_path = f"{config.cache_dir}/{flag}/block_training_quant_train"
        quant_val_cache_path = f"{config.cache_dir}/{flag}/block_training_quant_val"
        for path in [
            fp_train_cache_path,
            fp_val_cache_path,
            quant_train_cache_path,
            quant_val_cache_path,
        ]:
            if os.path.exists(path):
                shutil.rmtree(path)
    else:
        fp_train_cache_path = None
        fp_val_cache_path = None
        quant_train_cache_path = None
        quant_val_cache_path = None

    fp_train_inps = BlockTrainDataset(
        config.train_size,
        config.training_seqlen,
        model.model_config.hidden_size,
        config.batch_size,
        dtype,
        cache_path=fp_train_cache_path,
        off_load_to_disk=config.off_load_to_disk,
    )
    fp_val_inps = BlockTrainDataset(
        config.val_size,
        config.training_seqlen,
        model.model_config.hidden_size,
        config.batch_size,
        dtype,
        cache_path=fp_val_cache_path,
        off_load_to_disk=config.off_load_to_disk,
    )

    class Catcher(nn.Module):
        def __init__(self, module, dataset):
            super().__init__()
            self.module = module
            self.dataset = dataset
            self.index = 0
            self.attention_mask = None
            self.position_ids = None
            self.position_embeddings = None
            self.position_embeddings_global = None
            self.position_embeddings_local = None

        def forward(self, hidden_states, **kwargs):
            self.dataset.update_data(self.index, hidden_states.squeeze(0).to("cpu"))
            self.index += 1
            if self.attention_mask is None:
                self.attention_mask = kwargs.get("attention_mask")
            if self.position_ids is None:
                self.position_ids = kwargs.get("position_ids")
            if self.position_embeddings is None:
                pe = kwargs.get("position_embeddings")
                if pe is not None:
                    self.position_embeddings = tuple(
                        tensor.detach().cpu() if tensor is not None else None
                        for tensor in pe
                    )
            if self.position_embeddings_global is None:
                self.position_embeddings_global = kwargs.get(
                    "position_embeddings_global"
                )
            if self.position_embeddings_local is None:
                self.position_embeddings_local = kwargs.get("position_embeddings_local")
            raise ValueError

    layers[0] = Catcher(layers[0], fp_train_inps)
    iters = len(trainloader) // config.batch_size
    with torch.no_grad():
        for i in range(iters):
            samples = [
                trainloader[j][0]
                for j in range(i * config.batch_size, (i + 1) * config.batch_size)
            ]
            data = _collate_calibration_batch(samples, device_obj, pad_token_id)
            try:
                _invoke_model_forward(model.model, data)
            except ValueError:
                pass
    layers[0] = layers[0].module

    layers[0] = Catcher(layers[0], fp_val_inps)
    iters = len(valloader) // config.batch_size
    with torch.no_grad():
        for i in range(iters):
            samples = [
                valloader[j][0]
                for j in range(i * config.batch_size, (i + 1) * config.batch_size)
            ]
            data = _collate_calibration_batch(samples, device_obj, pad_token_id)
            try:
                _invoke_model_forward(model.model, data)
            except ValueError:
                pass
    attention_mask = layers[0].attention_mask
    position_ids = layers[0].position_ids
    position_embeddings = layers[0].position_embeddings if hasattr(layers[0], "position_embeddings") else None
    position_embeddings_global = (
        layers[0].position_embeddings_global if hasattr(layers[0], "position_embeddings_global") else None
    )
    position_embeddings_local = (
        layers[0].position_embeddings_local if hasattr(layers[0], "position_embeddings_local") else None
    )
    layers[0] = layers[0].module

    if attention_mask is not None:
        attention_mask_batch = _repeat_first_dim_if_needed(
            attention_mask, config.batch_size
        ).float()
    else:
        logger.info(
            "No attention mask caught from the first layer. Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    position_ids_batch = _repeat_first_dim_if_needed(position_ids, config.batch_size)
    position_embeddings = _repeat_position_embeddings_tuple(
        position_embeddings, config.batch_size, device_obj
    )
    position_embeddings_global = _repeat_first_dim_if_needed(
        position_embeddings_global, config.batch_size
    )
    if isinstance(position_embeddings_global, torch.Tensor):
        position_embeddings_global = position_embeddings_global.to(device_obj)
    position_embeddings_local = _repeat_first_dim_if_needed(
        position_embeddings_local, config.batch_size
    )
    if isinstance(position_embeddings_local, torch.Tensor):
        position_embeddings_local = position_embeddings_local.to(device_obj)

    layers[0] = layers[0].cpu()
    model.embedding = model.embedding.cpu()
    model.model_norm = model.model_norm.cpu()
    if hasattr(model, "rope"):
        model.rope = model.rope.cpu()
    if device_obj.type == "cuda":
        torch.cuda.empty_cache()

    if config.off_load_to_disk:
        shutil.copytree(fp_train_cache_path, quant_train_cache_path)
        shutil.copytree(fp_val_cache_path, quant_val_cache_path)
        quant_train_inps = BlockTrainDataset(
            config.train_size,
            config.training_seqlen,
            model.model_config.hidden_size,
            config.batch_size,
            dtype,
            cache_path=quant_train_cache_path,
            off_load_to_disk=config.off_load_to_disk,
        )
        quant_val_inps = BlockTrainDataset(
            config.val_size,
            config.training_seqlen,
            model.model_config.hidden_size,
            config.batch_size,
            dtype,
            cache_path=quant_val_cache_path,
            off_load_to_disk=config.off_load_to_disk,
        )
    else:
        quant_train_inps = BlockTrainDataset(
            config.train_size,
            config.training_seqlen,
            model.model_config.hidden_size,
            config.batch_size,
            dtype,
            cache_path=None,
            off_load_to_disk=False,
        )
        quant_val_inps = BlockTrainDataset(
            config.val_size,
            config.training_seqlen,
            model.model_config.hidden_size,
            config.batch_size,
            dtype,
            cache_path=None,
            off_load_to_disk=False,
        )
        for index, data in enumerate(fp_train_inps):
            quant_train_inps.update_data(index, data)
        for index, data in enumerate(fp_val_inps):
            quant_val_inps.update_data(index, data)

    loss_func = torch.nn.MSELoss()
    for block_index in tqdm(range(len(layers))):
        logger.info(f"=== Start quantize blocks {block_index}===")
        layer = layers[block_index].to(device_obj)
        qlayer = copy.deepcopy(layer)
        for name, module in qlayer.named_modules():
            if isinstance(module, nn.Linear):
                layer_prefix = wbits_config.get("layers_name_prefix", "")
                full_layer_name = (
                    f"{layer_prefix}.{block_index}.{name}"
                    if layer_prefix
                    else f"{block_index}.{name}"
                )
                wbits = wbits_config.get(full_layer_name, config.wbits)
                quantlinear = QuantLinear(module, wbits, config.group_size)
                set_op_by_name(qlayer, name, quantlinear)
        qlayer.to(device_obj)

        set_quant_state(qlayer, weight_quant=False)
        if config.epochs > 0:
            update_dataset(
                qlayer,
                fp_train_inps,
                device_obj,
                attention_mask_batch,
                position_ids_batch,
                position_embeddings=position_embeddings,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
            )
            update_dataset(
                qlayer,
                fp_val_inps,
                device_obj,
                attention_mask_batch,
                position_ids_batch,
                position_embeddings=position_embeddings,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
            )
        set_quant_state(qlayer, weight_quant=True)

        if config.epochs > 0:
            with torch.no_grad():
                qlayer.float()
            param = []
            assert config.quant_lr > 0 or config.weight_lr > 0
            total_training_iteration = (
                config.epochs * config.train_size / config.batch_size
            )
            quant_scheduler = None
            weight_scheduler = None
            quant_index = None
            weight_index = None
            param_group_index = 0
            if config.quant_lr > 0:
                set_quant_parameters(qlayer, True)
                param.append(
                    {"params": list(quant_parameters(qlayer)), "lr": config.quant_lr}
                )
                dummy_opt = torch.optim.AdamW(
                    [torch.tensor(0.0, device=device_obj)], lr=config.quant_lr
                )
                quant_scheduler = CosineAnnealingLR(
                    dummy_opt,
                    T_max=total_training_iteration,
                    eta_min=config.quant_lr / config.min_lr_factor,
                )
                quant_index = param_group_index
                param_group_index += 1
            else:
                set_quant_parameters(qlayer, False)
            if config.weight_lr > 0:
                set_weight_parameters(qlayer, True)
                param.append(
                    {"params": list(weight_parameters(qlayer)), "lr": config.weight_lr}
                )
                dummy_opt = torch.optim.AdamW(
                    [torch.tensor(0.0, device=device_obj)], lr=config.weight_lr
                )
                weight_scheduler = CosineAnnealingLR(
                    dummy_opt,
                    T_max=total_training_iteration,
                    eta_min=config.weight_lr / config.min_lr_factor,
                )
                weight_index = param_group_index
                param_group_index += 1
            else:
                set_weight_parameters(qlayer, False)

            optimizer = torch.optim.AdamW(param, weight_decay=config.wd)
            loss_scaler = NativeScalerWithGradNormCount(device_obj)
            best_val_loss = 1e6
            early_stop_flag = 0
            for epoch in range(config.epochs):
                loss_list = []
                norm_list = []
                start_time = time.time()
                for quant_inps, fp_inps in zip(quant_train_inps, fp_train_inps):
                    with torch.amp.autocast(device_type=device_obj.type):
                        input = quant_inps.to(device_obj)
                        label = fp_inps.to(device_obj)
                        quant_out = qlayer(
                            input,
                            attention_mask=attention_mask_batch,
                            position_ids=position_ids_batch,
                            position_embeddings=position_embeddings,
                            position_embeddings_global=position_embeddings_global,
                            position_embeddings_local=position_embeddings_local,
                        )[0]
                        reconstruction_loss = loss_func(
                            label, quant_out.to(torch.float32)
                        )
                        loss = reconstruction_loss
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NaN, stopping training")
                        break
                    loss_list.append(reconstruction_loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(
                        loss,
                        optimizer,
                        clip_grad=config.clip_grad,
                        parameters=trainable_parameters(qlayer),
                    )
                    norm_list.append(norm.cpu())
                    if quant_scheduler is not None and quant_index is not None:
                        quant_scheduler.step()
                        optimizer.param_groups[quant_index]["lr"] = (
                            quant_scheduler.get_last_lr()[0]
                        )
                    if weight_scheduler is not None and weight_index is not None:
                        weight_scheduler.step()
                        optimizer.param_groups[weight_index]["lr"] = (
                            weight_scheduler.get_last_lr()[0]
                        )

                val_loss_list = []
                with torch.no_grad():
                    for quant_inps, fp_inps in zip(quant_val_inps, fp_val_inps):
                        with torch.amp.autocast(device_type=device_obj.type):
                            input = quant_inps.to(device_obj)
                            label = fp_inps.to(device_obj)
                            quant_out = qlayer(
                                input,
                                attention_mask=attention_mask_batch,
                                position_ids=position_ids_batch,
                                position_embeddings=position_embeddings,
                                position_embeddings_global=position_embeddings_global,
                                position_embeddings_local=position_embeddings_local,
                            )[0]
                            reconstruction_loss = loss_func(
                                label, quant_out.to(torch.float32)
                            )
                        val_loss_list.append(reconstruction_loss.cpu())

                train_mean_num = min(len(loss_list), 64)
                loss_mean = torch.stack(loss_list)[-train_mean_num:].mean()
                val_loss_mean = torch.stack(val_loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                max_memory_mb = (
                    torch.cuda.max_memory_allocated(device_obj) / 1024**2
                    if device_obj.type == "cuda"
                    else 0.0
                )
                logger.info(
                    f"blocks {block_index} epoch {epoch} recon_loss:{loss_mean} val_loss:{val_loss_mean} "
                    f"norm:{norm_mean:.8f} max memory_allocated {max_memory_mb} time {time.time() - start_time}"
                )
                if val_loss_mean < best_val_loss:
                    best_val_loss = val_loss_mean
                else:
                    early_stop_flag += 1
                    if config.early_stop > 0 and early_stop_flag >= config.early_stop:
                        break
            optimizer.zero_grad()
            del optimizer

        qlayer = qlayer.half()
        quant_inplace(qlayer)
        set_quant_state(qlayer, weight_quant=False)

        if config.epochs > 0:
            update_dataset(
                qlayer,
                quant_train_inps,
                device_obj,
                attention_mask_batch,
                position_ids_batch,
                position_embeddings=position_embeddings,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
            )
            update_dataset(
                qlayer,
                quant_val_inps,
                device_obj,
                attention_mask_batch,
                position_ids_batch,
                position_embeddings=position_embeddings,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
            )
        layers[block_index] = qlayer.to("cpu")

        if config.real_quant:
            named_linears = get_named_linears(qlayer, QuantLinear)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scale.clamp(1e-4, 1e4).detach()
                zeros = module.weight_quantizer.zero_point.detach().to("cpu").round()
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0, -1).transpose(0, 1).contiguous()
                zeros = zeros.view(dim0, -1).transpose(0, 1).contiguous()
                if hasattr(module.weight_quantizer, "n_bits"):
                    bit_width = module.weight_quantizer.n_bits
                else:
                    bit_width = config.wbits
                q_linear = int_linear_real.QuantLinear(
                    bit_width,
                    group_size,
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                )
                q_linear.pack(module.cpu(), scales.float().cpu(), zeros.float().cpu())
                set_op_by_name(qlayer, name, q_linear)
        del layer
        if device_obj.type == "cuda":
            torch.cuda.empty_cache()

    if config.off_load_to_disk:
        for path in [
            fp_train_cache_path,
            fp_val_cache_path,
            quant_train_cache_path,
            quant_val_cache_path,
        ]:
            if path and os.path.exists(path):
                shutil.rmtree(path)

    if device_obj.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    model.model_config.use_cache = use_cache
    return model


def _block_ap_ring_buffer(
    model,
    config,
    pad_token_id,
    wbits_config,
    trainloader,
    valloader,
    device,
    logger,
    device_obj,
    dtype,
):
    layers = model.layers
    batch_size = config.batch_size
    total_train_samples = len(trainloader)
    if total_train_samples == 0:
        raise ValueError("Training loader is empty; cannot run ring-buffer QAT.")
    if total_train_samples % batch_size != 0:
        raise ValueError("train_size must be divisible by batch_size for ring-buffer QAT.")

    if hasattr(config, "train_size_in_memory"):
        train_mem_limit = config.train_size_in_memory
    else:
        train_mem_limit = config.train_size
    chunk_sample_limit = min(config.train_size, max(batch_size, train_mem_limit))
    if chunk_sample_limit % batch_size != 0:
        raise ValueError("train_size_in_memory must be divisible by batch_size.")

    total_train_batches = total_train_samples // batch_size
    chunk_batch_capacity = chunk_sample_limit // batch_size
    total_chunks = math.ceil(total_train_batches / chunk_batch_capacity)
    if logger:
        logger.info(
            "Processing %d calibration batches via ring buffer (%d samples per chunk).",
            total_train_batches,
            chunk_sample_limit,
        )

    fp_chunk_inps = BlockTrainDataset(
        chunk_sample_limit,
        config.training_seqlen,
        model.model_config.hidden_size,
        batch_size,
        dtype,
        cache_path=None,
        off_load_to_disk=False,
    )
    quant_chunk_inps = BlockTrainDataset(
        chunk_sample_limit,
        config.training_seqlen,
        model.model_config.hidden_size,
        batch_size,
        dtype,
        cache_path=None,
        off_load_to_disk=False,
    )

    val_total_samples = len(valloader)
    val_batches = val_total_samples // batch_size
    usable_val_samples = val_batches * batch_size

    attention_mask_batch = None
    position_ids_batch = None
    position_embeddings = None
    position_embeddings_global = None
    position_embeddings_local = None

    def _module_on_device(module) -> bool:
        try:
            param = next(module.parameters())
            return param.device == device_obj
        except StopIteration:
            try:
                buf = next(module.buffers())
                return buf.device == device_obj
            except StopIteration:
                return False

    def _move_prefix_to_device(upto_index: int) -> tuple[list[int], bool, bool, bool]:
        moved_indices: list[int] = []
        embedding_moved = False
        norm_moved = False
        rope_moved = False
        if hasattr(model, "embedding") and not _module_on_device(model.embedding):
            model.embedding = model.embedding.to(device=device_obj, dtype=dtype)
            embedding_moved = True
        if hasattr(model, "model_norm") and not _module_on_device(model.model_norm):
            model.model_norm = model.model_norm.to(device=device_obj, dtype=dtype)
            norm_moved = True
        if hasattr(model, "rope"):
            rope_obj = model.rope
            rope_device = rope_obj.device if hasattr(rope_obj, "device") else None
            if hasattr(rope_obj, "to") and rope_device != device_obj:
                model.rope = rope_obj.to(device_obj)
                rope_moved = True
        for idx in range(upto_index):
            module = layers[idx]
            if not _module_on_device(module):
                layers[idx] = module.to(device=device_obj, dtype=dtype)
                moved_indices.append(idx)
        return moved_indices, embedding_moved, norm_moved, rope_moved

    def _restore_prefix(moved_indices, embedding_moved, norm_moved, rope_moved):
        for idx in moved_indices:
            layers[idx] = layers[idx].to("cpu")
        if embedding_moved:
            model.embedding = model.embedding.to("cpu")
        if norm_moved:
            model.model_norm = model.model_norm.to("cpu")
        if rope_moved and hasattr(model, "rope"):
            model.rope = model.rope.to("cpu")

    class Catcher(nn.Module):
        def __init__(self, module, dataset):
            super().__init__()
            self.module = module
            self.dataset = dataset
            self.index = 0
            self.attention_mask = None
            self.position_ids = None
            self.position_embeddings = None
            self.position_embeddings_global = None
            self.position_embeddings_local = None

        def forward(self, hidden_states, **kwargs):
            self.dataset.update_data(self.index, hidden_states.squeeze(0).to("cpu"))
            self.index += 1
            if self.attention_mask is None:
                self.attention_mask = kwargs.get("attention_mask")
            if self.position_ids is None:
                self.position_ids = kwargs.get("position_ids")
            if self.position_embeddings is None:
                pe = kwargs.get("position_embeddings")
                if pe is not None:
                    self.position_embeddings = tuple(
                        tensor.detach().cpu() if tensor is not None else None
                        for tensor in pe
                    )
            if self.position_embeddings_global is None:
                self.position_embeddings_global = kwargs.get(
                    "position_embeddings_global"
                )
            if self.position_embeddings_local is None:
                self.position_embeddings_local = kwargs.get("position_embeddings_local")
            raise ValueError

    def _copy_dataset(src, dst):
        dst.set_active_batches(len(src))
        for idx in range(len(src)):
            dst.update_data(idx, src[idx])

    def _capture_batches(
        block_index,
        dataset,
        data_source,
        batch_start,
        num_batches,
    ):
        dataset.set_active_batches(num_batches)
        moved_prefix, embedding_moved, norm_moved, rope_moved = _move_prefix_to_device(
            block_index
        )
        layer_module = layers[block_index]
        if not _module_on_device(layer_module):
            layer_module = layer_module.to(device=device_obj, dtype=dtype)
            layers[block_index] = layer_module
        catcher = Catcher(layer_module, dataset)
        layers[block_index] = catcher
        try:
            with torch.no_grad():
                for local_batch in range(num_batches):
                    sample_start = (batch_start + local_batch) * batch_size
                    samples = [
                        data_source[sample_start + offset][0]
                        for offset in range(batch_size)
                    ]
                    data = _collate_calibration_batch(samples, device_obj, pad_token_id)
                    try:
                        _invoke_model_forward(model.model, data)
                    except ValueError:
                        pass
        finally:
            restored = catcher.module
            layers[block_index] = restored
            _restore_prefix(moved_prefix, embedding_moved, norm_moved, rope_moved)
        return catcher

    def _maybe_set_attention_metadata(catcher):
        nonlocal attention_mask_batch
        nonlocal position_ids_batch
        nonlocal position_embeddings
        nonlocal position_embeddings_global
        nonlocal position_embeddings_local
        if attention_mask_batch is None:
            attn = catcher.attention_mask
            if attn is not None:
                attention_mask_batch = _repeat_first_dim_if_needed(attn, batch_size).float()
            else:
                attention_mask_batch = None
        if position_ids_batch is None:
            pos_ids = catcher.position_ids
            if pos_ids is not None:
                position_ids_batch = _repeat_first_dim_if_needed(pos_ids, batch_size)
        if position_embeddings is None:
            position_embeddings = _repeat_position_embeddings_tuple(
                catcher.position_embeddings, batch_size, device_obj
            )
        if position_embeddings_global is None:
            global_pe = catcher.position_embeddings_global
            position_embeddings_global = _repeat_first_dim_if_needed(
                global_pe, batch_size
            )
            if isinstance(position_embeddings_global, torch.Tensor):
                position_embeddings_global = position_embeddings_global.to(device_obj)
        if position_embeddings_local is None:
            local_pe = catcher.position_embeddings_local
            position_embeddings_local = _repeat_first_dim_if_needed(
                local_pe, batch_size
            )
            if isinstance(position_embeddings_local, torch.Tensor):
                position_embeddings_local = position_embeddings_local.to(device_obj)

    loss_func = torch.nn.MSELoss()
    for block_index in range(len(layers)):
        if logger:
            logger.info(f"=== Start quantize blocks {block_index} (ring buffer)===")
        layer = layers[block_index].to(device=device_obj, dtype=dtype)
        layers[block_index] = layer
        qlayer = copy.deepcopy(layer)
        for name, module in qlayer.named_modules():
            if isinstance(module, nn.Linear):
                layer_prefix = wbits_config.get("layers_name_prefix", "")
                full_layer_name = (
                    f"{layer_prefix}.{block_index}.{name}"
                    if layer_prefix
                    else f"{block_index}.{name}"
                )
                wbits = wbits_config.get(full_layer_name, config.wbits)
                quantlinear = QuantLinear(module, wbits, config.group_size)
                set_op_by_name(qlayer, name, quantlinear)
        qlayer.to(device_obj)
        teacher_layer = copy.deepcopy(qlayer)
        teacher_layer.to(device_obj)
        set_quant_state(teacher_layer, weight_quant=False)
        for param in teacher_layer.parameters():
            param.requires_grad_(False)

        fp_val_inps = None
        quant_val_inps = None
        if usable_val_samples > 0:
            fp_val_inps = BlockTrainDataset(
                usable_val_samples,
                config.training_seqlen,
                model.model_config.hidden_size,
                batch_size,
                dtype,
                cache_path=None,
                off_load_to_disk=False,
            )
            quant_val_inps = BlockTrainDataset(
                usable_val_samples,
                config.training_seqlen,
                model.model_config.hidden_size,
                batch_size,
                dtype,
                cache_path=None,
                off_load_to_disk=False,
            )
            catcher = _capture_batches(
                block_index,
                fp_val_inps,
                valloader,
                batch_start=0,
                num_batches=val_batches,
            )
            _maybe_set_attention_metadata(catcher)
            _copy_dataset(fp_val_inps, quant_val_inps)
            update_dataset(
                teacher_layer,
                fp_val_inps,
                device_obj,
                attention_mask_batch,
                position_ids_batch,
                position_embeddings=position_embeddings,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
            )

        if config.epochs > 0:
            with torch.no_grad():
                qlayer.float()
                teacher_layer.float()
            set_quant_state(qlayer, weight_quant=True)
            param = []
            assert config.quant_lr > 0 or config.weight_lr > 0
            total_training_iteration = (
                config.epochs * total_train_batches
            )
            quant_scheduler = None
            weight_scheduler = None
            quant_index = None
            weight_index = None
            param_group_index = 0
            if config.quant_lr > 0:
                set_quant_parameters(qlayer, True)
                param.append(
                    {"params": list(quant_parameters(qlayer)), "lr": config.quant_lr}
                )
                dummy_opt = torch.optim.AdamW(
                    [torch.tensor(0.0, device=device_obj)], lr=config.quant_lr
                )
                quant_scheduler = CosineAnnealingLR(
                    dummy_opt,
                    T_max=total_training_iteration,
                    eta_min=config.quant_lr / config.min_lr_factor,
                )
                quant_index = param_group_index
                param_group_index += 1
            else:
                set_quant_parameters(qlayer, False)
            if config.weight_lr > 0:
                set_weight_parameters(qlayer, True)
                param.append(
                    {"params": list(weight_parameters(qlayer)), "lr": config.weight_lr}
                )
                dummy_opt = torch.optim.AdamW(
                    [torch.tensor(0.0, device=device_obj)], lr=config.weight_lr
                )
                weight_scheduler = CosineAnnealingLR(
                    dummy_opt,
                    T_max=total_training_iteration,
                    eta_min=config.weight_lr / config.min_lr_factor,
                )
                weight_index = param_group_index
                param_group_index += 1
            else:
                set_weight_parameters(qlayer, False)

            optimizer = torch.optim.AdamW(param, weight_decay=config.wd)
            loss_scaler = NativeScalerWithGradNormCount(device_obj)
            best_val_loss = 1e6
            early_stop_flag = 0

            for epoch in range(config.epochs):
                loss_list = []
                norm_list = []
                start_time = time.time()
                chunk_training_aborted = False
                batch_start = 0
                while batch_start < total_train_batches:
                    chunk_batches = min(
                        chunk_batch_capacity, total_train_batches - batch_start
                    )
                    catcher = _capture_batches(
                        block_index,
                        fp_chunk_inps,
                        trainloader,
                        batch_start=batch_start,
                        num_batches=chunk_batches,
                    )
                    _maybe_set_attention_metadata(catcher)
                    _copy_dataset(fp_chunk_inps, quant_chunk_inps)
                    update_dataset(
                        teacher_layer,
                        fp_chunk_inps,
                        device_obj,
                        attention_mask_batch,
                        position_ids_batch,
                        position_embeddings=position_embeddings,
                        position_embeddings_global=position_embeddings_global,
                        position_embeddings_local=position_embeddings_local,
                    )
                    for quant_inps, fp_inps in zip(quant_chunk_inps, fp_chunk_inps):
                        with torch.amp.autocast(device_type=device_obj.type):
                            input = quant_inps.to(device_obj)
                            label = fp_inps.to(device_obj)
                            quant_out = qlayer(
                                input,
                                attention_mask=attention_mask_batch,
                                position_ids=position_ids_batch,
                                position_embeddings=position_embeddings,
                                position_embeddings_global=position_embeddings_global,
                                position_embeddings_local=position_embeddings_local,
                            )[0]
                            reconstruction_loss = loss_func(
                                label, quant_out.to(torch.float32)
                            )
                            loss = reconstruction_loss
                        if not math.isfinite(loss.item()):
                            if logger:
                                logger.info("Loss is NaN, stopping training")
                            chunk_training_aborted = True
                            break
                        loss_list.append(reconstruction_loss.detach().cpu())
                        optimizer.zero_grad()
                        norm = loss_scaler(
                            loss,
                            optimizer,
                            clip_grad=config.clip_grad,
                            parameters=trainable_parameters(qlayer),
                        )
                        norm_list.append(norm.cpu())
                        if quant_scheduler is not None and quant_index is not None:
                            quant_scheduler.step()
                            optimizer.param_groups[quant_index]["lr"] = (
                                quant_scheduler.get_last_lr()[0]
                            )
                        if weight_scheduler is not None and weight_index is not None:
                            weight_scheduler.step()
                            optimizer.param_groups[weight_index]["lr"] = (
                                weight_scheduler.get_last_lr()[0]
                            )
                    if chunk_training_aborted:
                        break
                    batch_start += chunk_batches
                if chunk_training_aborted:
                    break

                val_loss_list = []
                if fp_val_inps is not None and quant_val_inps is not None:
                    with torch.no_grad():
                        for quant_inps, fp_inps in zip(quant_val_inps, fp_val_inps):
                            with torch.amp.autocast(device_type=device_obj.type):
                                input = quant_inps.to(device_obj)
                                label = fp_inps.to(device_obj)
                                quant_out = qlayer(
                                    input,
                                    attention_mask=attention_mask_batch,
                                    position_ids=position_ids_batch,
                                    position_embeddings=position_embeddings,
                                    position_embeddings_global=position_embeddings_global,
                                    position_embeddings_local=position_embeddings_local,
                                )[0]
                                reconstruction_loss = loss_func(
                                    label, quant_out.to(torch.float32)
                                )
                            val_loss_list.append(reconstruction_loss.cpu())

                train_mean_num = min(len(loss_list), 64) if loss_list else 0
                loss_mean = (
                    torch.stack(loss_list)[-train_mean_num:].mean()
                    if train_mean_num
                    else torch.tensor(0.0)
                )
                val_loss_mean = (
                    torch.stack(val_loss_list).mean()
                    if val_loss_list
                    else torch.tensor(0.0)
                )
                norm_mean = (
                    torch.stack(norm_list).mean() if norm_list else torch.tensor(0.0)
                )
                max_memory_mb = (
                    torch.cuda.max_memory_allocated(device_obj) / 1024**2
                    if device_obj.type == "cuda"
                    else 0.0
                )
                if logger:
                    logger.info(
                        f"blocks {block_index} epoch {epoch} recon_loss:{loss_mean} val_loss:{val_loss_mean} "
                        f"norm:{norm_mean:.8f} max memory_allocated {max_memory_mb} time {time.time() - start_time}"
                    )
                if val_loss_mean < best_val_loss:
                    best_val_loss = val_loss_mean
                else:
                    early_stop_flag += 1
                    if config.early_stop > 0 and early_stop_flag >= config.early_stop:
                        break
            optimizer.zero_grad()
            del optimizer

        qlayer = qlayer.half()
        quant_inplace(qlayer)
        set_quant_state(qlayer, weight_quant=False)
        del teacher_layer

        if config.real_quant:
            named_linears = get_named_linears(qlayer, QuantLinear)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scale.clamp(1e-4, 1e4).detach()
                zeros = module.weight_quantizer.zero_point.detach().to("cpu").round()
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0, -1).transpose(0, 1).contiguous()
                zeros = zeros.view(dim0, -1).transpose(0, 1).contiguous()
                if hasattr(module.weight_quantizer, "n_bits"):
                    bit_width = module.weight_quantizer.n_bits
                else:
                    bit_width = config.wbits
                q_linear = int_linear_real.QuantLinear(
                    bit_width,
                    group_size,
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                )
                q_linear.pack(module.cpu(), scales.float().cpu(), zeros.float().cpu())
                set_op_by_name(qlayer, name, q_linear)
        layers[block_index] = qlayer.to("cpu")
        if device_obj.type == "cuda":
            torch.cuda.empty_cache()

    if device_obj.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return model
