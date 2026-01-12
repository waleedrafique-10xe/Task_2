from unittest.mock import Mock, patch

import pytest

from GenAIQuant.model_preparer.model import Model
from GenAIQuant.model_preparer.preparer import ModelPreparer
from GenAIQuant.model_preparer.utils import ModelType


class TestModelPreparerBase:
    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing"""
        config = Mock()
        config.device = "cpu"
        config.quantization = Mock()
        config.quantization.pipeline = []
        return config

    def test_model_preparer_initialization(self, mock_config):
        """Test ModelPreparer initialization"""
        preparer = ModelPreparer("stas/tiny-random-llama-2", mock_config)

        assert preparer.model_id == "stas/tiny-random-llama-2"
        assert preparer.model_type == "llama"  # from model config
        assert preparer.config == mock_config
        assert preparer.quantization_config == mock_config.quantization
        assert preparer.model_config is not None
        assert preparer.architecture is not None

    def test_prepare_unsupported_model_type(self, mock_config):
        """Test preparing unsupported model type raises NotImplementedError"""
        # This is tricky to test without a real unsupported model
        # We can mock the model_type to be unsupported
        preparer = ModelPreparer("stas/tiny-random-llama-2", mock_config)
        preparer.model_type = "unsupported_model_type"

        with pytest.raises(
            NotImplementedError, match="model type unsupported_model_type not supported"
        ):
            preparer.prepare()


class TestLlamaModelPreparer(TestModelPreparerBase):
    @pytest.fixture
    def llama_preparer(self, mock_config):
        """Create a ModelPreparer for Llama model"""
        return ModelPreparer("stas/tiny-random-llama-2", mock_config)

    def test_prepare_llama_model(self, llama_preparer):
        """Test preparing a Llama model"""
        prepared_model = llama_preparer.prepare(fuse_layernorms=False)

        assert isinstance(prepared_model, Model)
        assert prepared_model.model_id == "stas/tiny-random-llama-2"
        assert prepared_model.model is not None
        assert prepared_model.layers is not None
        assert prepared_model.embedding is not None
        assert prepared_model.lm_head is not None
        assert prepared_model.meta is not None
        assert prepared_model.layernorm_fused is False

    def test_prepare_with_fused_layernorms(self, llama_preparer):
        """Test preparing model with fused layernorms"""
        with patch("GenAIQuant.model_preparer.preparer.fuse_rms_linear") as mock_fuse:
            prepared_model = llama_preparer.prepare(fuse_layernorms=True)

            assert isinstance(prepared_model, Model)
            assert prepared_model.layernorm_fused is True
            mock_fuse.assert_called_once()

    def test_prepare_handles_tied_embeddings(self, llama_preparer):
        """Test that prepare handles tied word embeddings"""
        prepared_model = llama_preparer.prepare()

        # Check that tie_word_embeddings is set to False
        assert prepared_model.model_config.tie_word_embeddings is False

    def test_model_eval_and_cache_settings(self, llama_preparer):
        """Test that model is set to eval mode and cache is disabled"""
        prepared_model = llama_preparer.prepare()

        assert prepared_model.model.training is False  # eval mode
        assert prepared_model.model.config.use_cache is False

    def test_prepare_sets_device_map_for_cpu(self, llama_preparer):
        """Test that device_map is None when device is CPU"""
        prepared_model = llama_preparer.prepare()

        # Since device is CPU in our mock config, device_map should be None
        assert prepared_model.device_map is None

    def test_embedding_weight_cloning(self, llama_preparer):
        """Test that embedding weights are properly cloned when tied"""
        prepared_model = llama_preparer.prepare()

        # Test that embedding and lm_head have different weight tensors
        embedding_weight_id = id(prepared_model.embedding.weight)
        lm_head_weight_id = id(prepared_model.lm_head.weight)

        # They should be different objects (cloned)
        assert embedding_weight_id != lm_head_weight_id

    def test_llama_has_no_vision_layers(self, llama_preparer):
        """Test that Llama model doesn't have vision layers"""
        prepared_model = llama_preparer.prepare()
        assert prepared_model.vision_layers is None


class TestGemma3ModelPreparer(TestModelPreparerBase):
    @pytest.fixture
    def gemma3_preparer(self, mock_config):
        """Create a ModelPreparer for Gemma3 model"""
        return ModelPreparer("tiny-random/gemma-3", mock_config)

    def test_gemma3_model_preparer_initialization(self, mock_config):
        """Test ModelPreparer initialization for Gemma3"""
        preparer = ModelPreparer("tiny-random/gemma-3", mock_config)

        assert preparer.model_id == "tiny-random/gemma-3"
        assert preparer.model_type == "gemma3"  # Should be gemma3, not llama
        assert preparer.config == mock_config
        assert preparer.quantization_config == mock_config.quantization
        assert preparer.model_config is not None
        assert preparer.architecture is not None

    def test_prepare_vlm_model(self, gemma3_preparer):
        """Test preparing a VLM model (Gemma3)"""
        prepared_model = gemma3_preparer.prepare()

        assert isinstance(prepared_model, Model)
        assert prepared_model.vision_layers is not None
        assert prepared_model.meta.model_type == ModelType.VLM

    def test_prepare_gemma3_model_components(self, gemma3_preparer):
        """Test preparing a Gemma3 model and verify all components"""
        prepared_model = gemma3_preparer.prepare(fuse_layernorms=False)

        assert isinstance(prepared_model, Model)
        assert prepared_model.model_id == "tiny-random/gemma-3"
        assert prepared_model.model is not None
        assert prepared_model.layers is not None
        assert prepared_model.embedding is not None
        assert prepared_model.lm_head is not None
        assert prepared_model.meta is not None
        assert prepared_model.meta.model_type == ModelType.VLM
        assert prepared_model.vision_layers is not None  # Gemma3 specific
        assert prepared_model.layernorm_fused is False

    def test_gemma3_vision_layers_structure(self, gemma3_preparer):
        """Test that Gemma3 vision layers are properly loaded"""
        prepared_model = gemma3_preparer.prepare()

        # Verify vision layers exist and have expected structure
        assert prepared_model.vision_layers is not None
        assert len(prepared_model.vision_layers) > 0

        # Verify meta has vision-specific configuration
        assert prepared_model.meta.vision_layers is not None
        assert prepared_model.meta.vision_layers_modules is not None
        assert len(prepared_model.meta.vision_layers_modules) == 4

    def test_gemma3_eval_and_cache_settings(self, gemma3_preparer):
        """Test that Gemma3 model is set to eval mode and cache is disabled"""
        prepared_model = gemma3_preparer.prepare()

        assert prepared_model.model.training is False  # eval mode
        assert prepared_model.model.config.use_cache is False

    def test_gemma3_device_map_for_cpu(self, gemma3_preparer):
        """Test that device_map is None when device is CPU for Gemma3"""
        prepared_model = gemma3_preparer.prepare()

        # Since device is CPU in our mock config, device_map should be None
        assert prepared_model.device_map is None


class TestModelComparisons(TestModelPreparerBase):
    def test_gemma3_vs_llama_differences(self, mock_config):
        """Test that Gemma3 and Llama models have different configurations"""
        llama_preparer = ModelPreparer("stas/tiny-random-llama-2", mock_config)
        gemma3_preparer = ModelPreparer("tiny-random/gemma-3", mock_config)

        llama_model = llama_preparer.prepare()
        gemma3_model = gemma3_preparer.prepare()

        # Different model types
        assert llama_model.meta.model_type != gemma3_model.meta.model_type

        # Llama shouldn't have vision layers, Gemma3 should
        assert llama_model.vision_layers is None
        assert gemma3_model.vision_layers is not None

        # Different layer paths
        assert llama_model.meta.layers != gemma3_model.meta.layers
        assert llama_model.meta.embedding != gemma3_model.meta.embedding

    def test_both_models_can_be_prepared_in_same_session(self, mock_config):
        """Test that both models can be prepared in the same test session"""
        # This ensures there are no global state conflicts
        llama_preparer = ModelPreparer("stas/tiny-random-llama-2", mock_config)
        gemma3_preparer = ModelPreparer("tiny-random/gemma-3", mock_config)

        llama_model = llama_preparer.prepare()
        gemma3_model = gemma3_preparer.prepare()

        assert isinstance(llama_model, Model)
        assert isinstance(gemma3_model, Model)
        assert llama_model.model_id != gemma3_model.model_id
        assert llama_model.meta != gemma3_model.meta
