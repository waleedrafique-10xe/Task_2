from typing import TYPE_CHECKING

# from accelerate import infer_auto_device_map # TODO: Bring this back evantually
from torch import nn
from transformers import AutoConfig
from transformers.modeling_utils import PreTrainedModel

from GenAIQuant.config.constants import PipelineModules
from GenAIQuant.logger import logger
from GenAIQuant.utils import get_nested_attr

from .graph_manipulation import fuse_rms_linear
from .model import Model
from .model_name_mapping import SUPPORTED_ARCHITECTURES
from .utils import ModelType

if TYPE_CHECKING:
    from GenAIQuant.config import Config


class ModelPreparer:
    """
    Prepares a model for quantization by extracting key components and generating an
    exported program.

    Creates and instance of `Model`

    [WARNING]: Suports preparation of LLMS only for now.

    Attributes:
        model (PreTrainedModel): The loaded pre-trained language model.
        model_config (PretrainedConfig): Configuration of the model.
        quantization_config (QuantizationConfig): Configuration for quantization.
        model_type (str): Type of the model.
        architecture (str): Architecture of the model.
    """

    def __init__(self, model_id: str, config: "Config"):
        """
        Initialize a ModelPreparer instance.

        Args:
            model_id (str): Identifier of the pre-trained model to load.
            config (QuantizationConfig): Configuration for quantization.
        """
        self.model_id = model_id
        self.model_config = AutoConfig.from_pretrained(model_id)
        self.model_type = self.model_config.model_type

        self.config = config
        self.quantization_config = config.quantization

        architectures = self.model_config.architectures
        if not architectures:
            # Some configs omit `architectures`; fall back to model type.
            architectures = [self.model_type]
        self.architecture = architectures[0]

    def prepare(self, fuse_layernorms: bool = True):
        """
        Prepare the model for quantization by:
        1. Identifying decoder layers and embedding
        2. Optionally fusing layer norms

        Args:
            fuse_layernorms (bool, optional):
                Whether to fuse layer norm and linear layers.
                Defaults to True.

        Returns:
            Model: A prepared model based on model specific config.

        Raises:
            Exception: If the model type is not supported or layers/embedding cannot be
            found.
        """

        # checking for supported model architectures
        # if that fails then throw Exception

        model_meta = SUPPORTED_ARCHITECTURES.get(self.model_type, None)
        if model_meta is None:
            raise NotImplementedError(
                f"model type {self.model_type} not currently supported"
            )

        model_class = model_meta.architecture_class

        model: PreTrainedModel = model_class.from_pretrained(
            self.model_id,
            torch_dtype=self.model_config.torch_dtype,
            low_cpu_mem_usage=True,
        )
        model.eval()
        model.config.use_cache = False
        # ensure cache is set to False
        self.model_config = model.config
        model_config = model.config

        logger.info(
            f"Successfully loaded model with model_id: {self.model_id} "
            f"into class {model_class}"
        )

        layers_name = model_meta.layers
        layers = get_nested_attr(model, layers_name)
        logger.info(f"Localized model decoder layers from {layers_name}")

        embedding_name = model_meta.embedding
        embedding = get_nested_attr(model, embedding_name)
        logger.info(f"Localized model embedding layer from {embedding_name}")

        rope_name = model_meta.rope
        rope = get_nested_attr(model, rope_name)

        lm_head_name = model_meta.lm_head
        lm_head = get_nested_attr(model, lm_head_name)

        model_norm_name = model_meta.model_norm
        model_norm = get_nested_attr(model, model_norm_name)

        logger.info(f"Localized model lm_head from {lm_head_name}")

        vision_layers = None
        if model_meta.model_type == ModelType.VLM:
            vision_layers_name = model_meta.vision_layers
            assert vision_layers_name is not None

            vision_layers = get_nested_attr(model, vision_layers_name)
            logger.info(
                f"Localized model (vlm) encoder layers from {vision_layers_name}"
            )

        if model_config.tie_word_embeddings:
            # embedding layer and lm_head are using the same weight tensors
            # separate embeddings and lm_head weight tensors
            embedding.weight = nn.Parameter(embedding.weight.clone())
            self.model_config.tie_word_embeddings = False

            if model_meta.model_type == ModelType.VLM:
                self.model_config.text_config.tie_word_embeddings = False

            logger.info(f"Untied word embeddiings from {model._tied_weights_keys}")

        if fuse_layernorms:
            fuse_rms_linear(model, SUPPORTED_ARCHITECTURES[self.model_type])
            logger.info("Successfully fused RMS Norm into Linear Layers")

        if PipelineModules.PTQ in self.quantization_config.pipeline:
            # add QDQ Ops
            pass

        return Model(
            model_id=self.model_id,
            model=model,
            layers=layers,
            embedding=embedding,
            rope=rope,
            model_norm=model_norm,
            lm_head=lm_head,
            model_meta=model_meta,
            model_config=model_config,
            layernorm_fused=fuse_layernorms,
            vision_layers=vision_layers,
            device=self.config.device,  # TODO: validate correctness of config.device
            # in config parser
        )
