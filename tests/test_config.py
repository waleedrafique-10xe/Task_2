"""Comprehensive test suite for GenAIQuant configuration classes and enums.

This module contains pytest test cases for all configuration components
including enums, configuration classes, validation, and edge cases.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from GenAIQuant.algorithms import bit_allocation, outlier_reduction, ptq, qat
from GenAIQuant.config.algorithm_configs import (
    BitAllocationConfig,
    OutlierReductionConfig,
    PtqConfig,
    QatConfig,
)
from GenAIQuant.config.config import (
    Config,
    DatasetType,
    EvaluationConfig,
    PipelineModules,
    QuantizationConfig,
)


class TestEnums:
    """Test cases for all enum classes."""

    def test_dataset_type_values(self):
        """Test DatasetType enum values."""
        assert DatasetType.HUGGINGFACE == "huggingface"
        assert DatasetType.LOCAL == "local"
        assert len(DatasetType) == 2

    def test_pipeline_modules_values(self):
        """Test PipelineModules enum values."""
        assert PipelineModules.OUTLIER_REDUCTION == "outlier_reduction"
        assert PipelineModules.BIT_ALLOCATION == "bit_allocation"
        assert PipelineModules.PTQ == "ptq"
        assert PipelineModules.QAT == "qat"
        assert len(PipelineModules) == 4


class TestEvaluationConfig:
    """Test cases for EvaluationConfig class."""

    def test_creation_minimal(self):
        """Test creating EvaluationConfig with minimal required fields."""
        config = EvaluationConfig()

        assert config.task == "wikitext"
        assert config.batchsize == 1
        assert config.seqlength == 512
        assert config.num_samples is None
        assert config.output_dir == "evaluation_output"
        assert not config.save_output

    def test_creation_full(self):
        """Test creating EvaluationConfig with all fields."""
        config = EvaluationConfig(
            task=["wikitext", "lambada"],
            batchsize=32,
            seqlength=1024,
            num_samples=100,
            output_dir="eval_results",
            save_output=True,
        )

        assert config.task == ["wikitext", "lambada"]
        assert config.batchsize == 32
        assert config.seqlength == 1024
        assert config.num_samples == 100
        assert config.output_dir == "eval_results"
        assert config.save_output is True

    def test_creation_with_defaults(self):
        """Test creating EvaluationConfig with default values."""
        config = EvaluationConfig()

        assert config.task == "wikitext"
        assert config.batchsize == 1
        assert config.seqlength == 512
        assert config.num_samples is None
        assert config.output_dir == "evaluation_output"
        assert config.save_output is False

    def test_single_task_string(self):
        """Test EvaluationConfig with string task."""
        config = EvaluationConfig(task="lambada")
        assert config.task == "lambada"

    def test_validation_batchsize(self):
        """Test validation of batchsize field."""
        # Valid positive number
        config = EvaluationConfig(batchsize=16)
        assert config.batchsize == 16

        # Invalid zero or negative number
        with pytest.raises(ValidationError):
            EvaluationConfig(batchsize=0)

        with pytest.raises(ValidationError):
            EvaluationConfig(batchsize=-5)

    def test_validation_seqlength(self):
        """Test validation of seqlength field."""
        # Valid positive number
        config = EvaluationConfig(seqlength=2048)
        assert config.seqlength == 2048

        # Invalid zero or negative number
        with pytest.raises(ValidationError):
            EvaluationConfig(seqlength=0)

        with pytest.raises(ValidationError):
            EvaluationConfig(seqlength=-10)

    def test_num_samples_nullable(self):
        """Test that num_samples can be None."""
        config = EvaluationConfig(num_samples=None)
        assert config.num_samples is None

        # Also test with a valid positive number
        config = EvaluationConfig(num_samples=50)
        assert config.num_samples == 50


class TestOutlierReductionConfig:
    """Test cases for OutlierReductionConfig class."""

    def test_default(self):
        """Test OutlierReductionConfig with default values."""
        config = OutlierReductionConfig()
        assert config.algorithm == outlier_reduction.SpinQuant

    def test_custom_with_class(self):
        """Test OutlierReductionConfig with custom algorithm."""
        config = OutlierReductionConfig(algorithm=outlier_reduction.SpinQuant)
        assert config.algorithm == outlier_reduction.SpinQuant

    def test_custom_with_string(self):
        """Test OutlierReductionConfig with custom algorithm."""
        config = OutlierReductionConfig(algorithm="SpinQuant")
        assert config.algorithm == outlier_reduction.SpinQuant

    def test_default_classmethod(self):
        """Test OutlierReductionConfig.default() class method."""
        config = OutlierReductionConfig.default()
        assert isinstance(config, OutlierReductionConfig)
        assert config.algorithm == outlier_reduction.SpinQuant

    def test_error_case(self):
        """Test OutlierReductionConfig with invalid algorithm."""
        with pytest.raises(ValueError):
            OutlierReductionConfig(algorithm="InvalidAlgorithm")


class TestBitAllocationConfig:
    """Test cases for BitAllocationConfig class."""

    def test_default(self):
        """Test BitAllocationConfig with default values."""
        config = BitAllocationConfig()
        assert config.algorithm == bit_allocation.ShortGpt
        assert config.target_avg_bitwidth == 3
        assert config.bitwidth_options == [2, 3, 4]
        assert config.dataset_name == "wikitext"
        assert config.num_samples == 128
        assert config.seq_len == 1024
        assert not config.use_jacobian

    def test_custom_with_class(self):
        """Test BitAllocationConfig with custom algorithm."""
        config = BitAllocationConfig(algorithm=bit_allocation.Uniform)
        assert config.algorithm == bit_allocation.Uniform

    def test_custom_with_string(self):
        """Test BitAllocationConfig with custom algorithm."""
        config = BitAllocationConfig(algorithm="Uniform")
        assert config.algorithm == bit_allocation.Uniform

    def test_default_classmethod(self):
        """Test BitAllocationConfig.default() class method."""
        config = BitAllocationConfig.default()
        assert isinstance(config, BitAllocationConfig)
        assert config.algorithm == bit_allocation.ShortGpt
        assert config.target_avg_bitwidth == 3
        assert config.bitwidth_options == [2, 3, 4]
        assert config.dataset_name == "wikitext"
        assert config.num_samples == 128
        assert config.seq_len == 1024
        assert not config.use_jacobian

    def test_creation_from_string(self):
        test_string = {
            "dataset_name": "Some name",
            "bitwidth_options": ["1", "2", "3"],
        }

        config = BitAllocationConfig(**test_string)
        assert isinstance(config.bitwidth_options, list)
        assert isinstance(config.bitwidth_options[0], int)

    def test_error_case(self):
        """Test BitAllocationConfig with invalid algorithm."""
        with pytest.raises(ValueError):
            BitAllocationConfig(algorithm="InvalidAlgorithm")

        with pytest.raises(ValueError):
            BitAllocationConfig(bitwidth_options=[])

        with pytest.raises(ValueError):
            BitAllocationConfig(bitwidth_options=[1, 2, 53])
            BitAllocationConfig(bitwidth_options=[1, 2, -1])
            BitAllocationConfig(bitwidth_options=[1, 2, 0])


class TestPtqConfig:
    """Test cases for PtqConfig class."""

    def test_default(self):
        """Test PtqConfig with default values."""
        config = PtqConfig()
        assert config.algorithm == ptq.Gptq

    def test_custom_with_string(self):
        """Test PtqConfig with custom algorithm."""
        config = PtqConfig(algorithm="Awq")
        assert config.algorithm == ptq.Awq

    def test_default_classmethod(self):
        """Test PtqConfig.default() class method."""
        config = PtqConfig.default()
        assert isinstance(config, PtqConfig)
        assert config.algorithm == ptq.Gptq

    def test_error_case(self):
        """Test PtqConfig with invalid algorithm."""
        with pytest.raises(ValueError):
            PtqConfig(algorithm="InvalidAlgorithm")


class TestQatConfig:
    """Test cases for QatConfig class."""

    def test_default(self):
        """Test QatConfig with default values."""
        config = QatConfig()
        assert config.algorithm == qat.EfficientQat

    def test_custom_with_string(self):
        """Test QatConfig with custom algorithm."""
        config = QatConfig(algorithm="EfficientQat")
        assert config.algorithm == qat.EfficientQat

    def test_default_classmethod(self):
        """Test QatConfig.default() class method."""
        config = QatConfig.default()
        assert isinstance(config, QatConfig)
        assert config.algorithm == qat.EfficientQat

    def test_error_case(self):
        """Test QatConfig with invalid algorithm."""
        with pytest.raises(ValueError):
            QatConfig(algorithm="InvalidAlgorithm")


class TestQuantizationConfig:
    """Test cases for QuantizationConfig class."""

    def test_default(self):
        """Test QuantizationConfig with default values."""
        config = QuantizationConfig()

        expected_pipeline = [
            PipelineModules.OUTLIER_REDUCTION,
            PipelineModules.BIT_ALLOCATION,
            PipelineModules.QAT,
        ]
        assert config.pipeline == expected_pipeline
        assert isinstance(config.outlier_reduction, OutlierReductionConfig)
        assert isinstance(config.bit_allocation, BitAllocationConfig)
        assert isinstance(config.ptq, PtqConfig)
        assert isinstance(config.qat, QatConfig)

    def test_custom_pipeline(self):
        """Test QuantizationConfig with custom pipeline."""
        custom_pipeline = [PipelineModules.PTQ, PipelineModules.QAT]
        config = QuantizationConfig(pipeline=custom_pipeline)

        assert config.pipeline == custom_pipeline

    def test_custom_configs(self):
        """Test QuantizationConfig with custom sub-configurations."""
        custom_outlier = OutlierReductionConfig(algorithm="SpinQuant")
        custom_bit_alloc = BitAllocationConfig(algorithm="Uniform")
        custom_ptq = PtqConfig(algorithm="Awq")
        custom_qat = QatConfig(algorithm="EfficientQat")

        config = QuantizationConfig(
            outlier_reduction=custom_outlier,
            bit_allocation=custom_bit_alloc,
            ptq=custom_ptq,
            qat=custom_qat,
        )

        assert config.outlier_reduction.algorithm == outlier_reduction.SpinQuant
        assert config.bit_allocation.algorithm == bit_allocation.Uniform
        assert config.ptq.algorithm == ptq.Awq
        assert config.qat.algorithm == qat.EfficientQat

    def test_default_classmethod(self):
        """Test QuantizationConfig.default() class method."""
        config = QuantizationConfig.default()
        assert isinstance(config, QuantizationConfig)


class TestConfig:
    """Test cases for main Config class."""

    def test_creation_minimal(self):
        """Test creating Config with minimal required fields."""
        evaluation_config = EvaluationConfig()

        config = Config(evaluation=evaluation_config)

        assert config.output_dir == Path("output")
        assert isinstance(config.quantization, QuantizationConfig)
        assert config.evaluation == evaluation_config

    def test_creation_full(self):
        """Test creating Config with all fields."""
        evaluation_config = EvaluationConfig(
            task=["wikitext", "lambada"],
            batchsize=32,
            seqlength=1024,
            num_samples=100,
            output_dir="eval_results",
            save_output=True,
        )

        quantization_config = QuantizationConfig(
            pipeline=[PipelineModules.PTQ, PipelineModules.QAT]
        )

        config = Config(
            output_dir=Path("/custom/output"),
            quantization=quantization_config,
            evaluation=evaluation_config,
        )

        assert config.output_dir == Path("/custom/output")
        assert config.quantization == quantization_config
        assert config.evaluation == evaluation_config

    def test_from_dict_minimal(self):
        """Test Config.from_dict() with minimal data."""
        data = {}

        config = Config.from_dict(data)

        assert isinstance(config, Config)
        assert config.evaluation is None

    def test_from_dict_full(self):
        """Test Config.from_dict() with comprehensive data."""
        data = {
            "output_dir": "/home/user/output",
            "evaluation": {
                "task": ["wikitext", "lambada"],
                "batchsize": 32,
                "seqlength": 1024,
                "num_samples": 100,
                "output_dir": "eval_results",
                "save_output": True,
            },
            "quantization": {
                "pipeline": ["bit_allocation", "qat"],
                "bit_allocation": {"algorithm": "Uniform"},
                "qat": {"algorithm": "EfficientQat"},
            },
        }

        config = Config.from_dict(data)
        assert config.evaluation is not None
        assert config.output_dir == Path("/home/user/output")
        assert config.evaluation.task == ["wikitext", "lambada"]
        assert config.evaluation.num_samples == 100
        assert config.evaluation.batchsize == 32
        assert config.quantization.pipeline == [
            PipelineModules.BIT_ALLOCATION,
            PipelineModules.QAT,
        ]
        assert config.quantization.bit_allocation.algorithm == bit_allocation.Uniform

    def test_model_dump(self):
        """Test Config model serialization."""

        config = Config(evaluation=None)
        dumped = config.model_dump()

        assert isinstance(dumped, dict)
        assert "output_dir" in dumped
        assert "quantization" in dumped
        assert "evaluation" in dumped


class TestConfigIntegration:
    """Integration tests for configuration classes."""

    def test_main_example_from_file(self):
        """Test the example configuration from the main block."""
        main_config = {
            "output_dir": "/home/mahd/output",
            "evaluation": {
                "batchsize": 32,
                "num_samples": 4,
            },
        }

        config = Config.from_dict(main_config)

        assert config.output_dir == Path("/home/mahd/output")
        assert config.evaluation is not None
        assert config.evaluation.batchsize == 32
        assert config.evaluation.num_samples == 4

    def test_round_trip_serialization(self):
        """Test serialization and deserialization round trip."""
        original_data = {
            "output_dir": "/test/output",
            "evaluation": {
                "task": ["wikitext", "lambada"],
                "batchsize": 32,
                "seqlength": 1024,
                "num_samples": 100,
                "output_dir": "eval_results",
                "save_output": True,
            },
            "quantization": {
                "pipeline": ["outlier_reduction", "ptq"],
                "outlier_reduction": {"algorithm": "SpinQuant"},
                "ptq": {"algorithm": "Awq"},
            },
        }

        # Create config from dict
        config = Config.from_dict(original_data)

        # Serialize to dict
        serialized = config.model_dump()

        # Create new config from serialized data
        config2 = Config.from_dict(serialized)

        # Verify they're equivalent
        assert config.evaluation is not None and config2.evaluation is not None
        assert config.output_dir == config2.output_dir
        assert config.evaluation == config2.evaluation
        assert config.quantization.pipeline == config2.quantization.pipeline
