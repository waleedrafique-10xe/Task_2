# my_models/custom_model.py
import torch
from GenAIQuant.logger import logger
from .base import BaseModel
from qwen_vl_utils import process_vision_info

DEFAULT_GENERATION_KWARGS = {
    "do_sample": False,
    "max_new_tokens": 2048,
    "repetition_penalty": 1.05,
}

DEFAULT_FALLBACK_GENERATION_KWARGS = {
    "do_sample": False,
    "max_new_tokens": 2048,
}


class CustomVLM(BaseModel):
    """
    Wrapper for a Visual-Language Model (VLM) to handle prompt formatting,
    message preprocessing, and inference generation for both chat-based
    and generic generation templates.

    This class abstracts away preprocessing logic (chat template vs. generation
    template) and provides a unified interface to run inference given text/image
    inputs. It supports:
        - Chat-style prompts (role-based)
        - Image+text generation templates
        - Automatic device and dtype management
        - Safe inference without gradient computation

    Attributes:
        temperature (float): Sampling temperature for generation.
        processor (object): Preprocessing pipeline (e.g., tokenizer, feature extractor).
        model (nn.Module): Pretrained model used for inference.
    """

    def __init__(
        self,
        model,
        procesor,
        temperature=0.7,
        generation_config=None,
        fallback_generation_config=None,
    ):
        """
        Initialize the CustomVLM wrapper.

        Args:
            model (nn.Module): The pretrained VLM model to wrap.
            processor (object): Preprocessing pipeline for tokenization,
                chat template application, and image handling.
            temperature (float, optional): Sampling temperature for generation. Defaults to 0.7 but not being used currently.
        """
        self.temperature = temperature  # not being used currently
        self.processor = procesor
        self.model = model
        self.generation_kwargs = {
            **DEFAULT_GENERATION_KWARGS,
            **(generation_config or {}),
        }
        self.fallback_generation_kwargs = {
            **DEFAULT_FALLBACK_GENERATION_KWARGS,
            **(fallback_generation_config or {}),
        }

    def generate_inner(self, message, dataset=None):
        """
        Generate a model response given a multi-modal message.
        A must to have function for custom model to be added in VLMEvalKit.

        Args:
            message (list[dict]): Sequence of message elements, where each element is:
                {
                    "type": "text" | "image",
                    "value": str  # text string or image path/URL
                }
            dataset (optional): Dataset reference for context (unused by default).

        Returns:
            str: The decoded model output string.
        """

        inputs, input_len = self._prepare_model_inputs(message)
        try:
            sequences = self._run_generation(inputs, self.generation_kwargs)
        except Exception as err:
            if not self._is_recoverable_error(err):
                raise
            logger.warning(
                "Primary generation failed with %s; retrying with fallback settings.",
                err,
            )
            torch.cuda.empty_cache()
            fallback_inputs, fallback_input_len = self._prepare_model_inputs(message)
            sequences = self._run_generation(
                fallback_inputs, self.fallback_generation_kwargs
            )
            input_len = fallback_input_len

        generation = sequences[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        torch.cuda.empty_cache()
        return decoded

    def _prepare_model_inputs(self, message):
        if self.has_chat_template(self.processor):
            inputs = self.chat_template(message)
        else:
            inputs = self.generation_template(message)

        if hasattr(inputs, "to"):
            inputs = inputs.to(
                device=self._resolve_device(), dtype=self._resolve_dtype()
            )

        plain_inputs = self._clone_structure(dict(inputs))
        return plain_inputs, plain_inputs["input_ids"].shape[-1]

    def _run_generation(self, inputs, generation_kwargs):
        model_kwargs = self._clone_structure(inputs)
        with torch.inference_mode():
            with torch.no_grad():
                return self.model.generate(**model_kwargs, **generation_kwargs)

    def _clone_structure(self, value):
        if torch.is_tensor(value):
            return value.clone()
        if isinstance(value, dict):
            return {k: self._clone_structure(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._clone_structure(v) for v in value)
        return value

    def _resolve_device(self):
        try:
            return self.model.device
        except AttributeError:
            try:
                return next(self.model.parameters()).device
            except (StopIteration, AttributeError):
                return torch.device("cpu")

    def _resolve_dtype(self):
        try:
            return next(self.model.parameters()).dtype
        except (StopIteration, AttributeError):
            return getattr(self.model, "dtype", torch.float32)

    def _is_recoverable_error(self, err: Exception) -> bool:
        recoverable = (RuntimeError,)
        cuda_error = getattr(torch.cuda, "CudaError", None)
        if cuda_error is not None:
            recoverable = recoverable + (cuda_error,)
        accelerator_error = getattr(torch.cuda, "AcceleratorError", None)
        if accelerator_error is not None:
            recoverable = recoverable + (accelerator_error,)
        return isinstance(err, recoverable)

    def has_chat_template(self, processor) -> bool:
        """
        Check whether the processor contains a chat template.

        Args:
            processor (object): The processor to inspect.

        Returns:
            bool: True if a chat template exists, False otherwise.
        """

        try:
            tokenizer = getattr(processor, "tokenizer", None)
            return bool(getattr(tokenizer, "chat_template", None))
        except Exception:
            return False

    def chat_template(self, message):
        """
        Build input tensors from a message using a chat-style prompt.

        Args:
            message (list[dict]): List of message elements containing
                text and/or image entries.

        Returns:
            dict: Tokenized and tensorized model inputs ready for inference.
        """

        ret = []
        if hasattr(self, "system_prompt") and self.system_prompt is not None:
            ret = [
                dict(
                    role="system", content=[dict(type="text", text=self.system_prompt)]
                )
            ]
        content = []
        min_pixels = (16 * 28) ** 2
        max_pixels = 28 ** 4
        for m in message:
            if m["type"] == "text":
                content.append(dict(type="text", text=m["value"]))
            elif m["type"] == "image":
                content.append(dict(type="image", min_pixels=min_pixels, max_pixels=max_pixels, image=m["value"]))
        ret.append(dict(role="user", content=content))

        inputs = self.processor.apply_chat_template(
            ret,
            add_generation_prompt=True,
            tokenize=False,
            return_dict=True,
            return_tensors="pt",
        )
        image_inputs, video_inputs = process_vision_info(ret)
        inputs = self.processor(
            text=[inputs],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._resolve_device(), dtype=self._resolve_dtype())
        return inputs

    def generation_template(self, message):
        """
        Build input tensors from a message using a generic generation template.

        Args:
            message (list[dict]): List of message elements containing
                text and/or image entries.

        Returns:
            dict: Tokenized and tensorized model inputs ready for inference.
        """

        for m in message:
            if m["type"] == "text":
                prompt = m["value"]
            elif m["type"] == "image":
                url = m["value"]
        prompt = ("<start_of_image> {}").format(prompt)
        image = self.load_image(url)
        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self._resolve_device(), dtype=self._resolve_dtype())
        return inputs

    def load_image(self, path_or_url):
        """
        Load an image from a local path or remote URL.

        Args:
            path_or_url (str): Path to the local image file or full URL.

        Returns:
            PIL.Image.Image: Loaded image object.

        Raises:
            requests.exceptions.RequestException: If downloading from a URL fails.
            FileNotFoundError: If the local image file does not exist.
        """

        from PIL import Image
        import requests

        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            response = requests.get(path_or_url, stream=True).raw()
            return Image.open(BytesIO(response.content))
        else:
            return Image.open(path_or_url)
