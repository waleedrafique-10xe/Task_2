import pytest
import torch

from GenAIQuant.algorithms import outlier_reduction
from GenAIQuant.algorithms.base import Algorithm
from GenAIQuant.config import Config, PipelineModules
from GenAIQuant.pipeline.pipeline import Pipeline, PipelineBuilder


class TestPipelineBuilder:
    @pytest.fixture
    def mock_algorithm(self):
        class MockAlgorithm(Algorithm):
            def forward(self):
                raise NotImplementedError("not implemented")

        return MockAlgorithm

    @pytest.fixture
    def config_with_modules(self, mock_algorithm, monkeypatch):
        cfg = Config()

        monkeypatch.setattr(
            cfg.quantization.outlier_reduction, "algorithm", lambda c: mock_algorithm
        )
        monkeypatch.setattr(
            cfg.quantization.bit_allocation, "algorithm", lambda c: mock_algorithm
        )
        monkeypatch.setattr(cfg.quantization.ptq, "algorithm", lambda c: mock_algorithm)
        monkeypatch.setattr(cfg.quantization.qat, "algorithm", lambda c: mock_algorithm)

        cfg.quantization.pipeline = [
            PipelineModules.OUTLIER_REDUCTION,
            PipelineModules.BIT_ALLOCATION,
            PipelineModules.PTQ,
            PipelineModules.QAT,
        ]
        return cfg

    def test_add_method_adds_algorithm(self, mock_algorithm):
        builder = PipelineBuilder()
        builder.add(mock_algorithm)
        assert mock_algorithm in builder.pipeline

    def test_build_returns_pipeline_instance(self, mock_algorithm):
        builder = PipelineBuilder()
        builder.add(mock_algorithm)
        pipeline = builder.build()
        assert isinstance(pipeline, Pipeline)
        assert pipeline.pipeline == [mock_algorithm]

    def test_build_from_config_adds_correct_algorithms(self, config_with_modules):
        builder = PipelineBuilder()
        pipeline = builder.build_from_config(config_with_modules)
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.pipeline) == 4

    def test_pipeline_run_raises_not_implemented(self, mock_algorithm):
        pipeline = Pipeline([mock_algorithm])
        with pytest.raises(NotImplementedError):
            pipeline.run(torch.nn.Module())

    def test_add_invalid_algorithm_str_raises_value_error(self):
        builder = PipelineBuilder()
        with pytest.raises(
            ValueError, match="Algorithm `NonExistentAlgo` is not available."
        ):
            builder.add("NonExistentAlgo")

    def test_add_valid_algorithm_class(self):
        class DummyAlgorithm(Algorithm):
            def forward(self):
                raise NotImplementedError()

        builder = PipelineBuilder()
        builder.add(DummyAlgorithm)
        assert builder.pipeline[-1] is DummyAlgorithm

    def test_add_invalid_type_raises_type_error(self):
        builder = PipelineBuilder()
        with pytest.raises(
            TypeError,
            match="Algorithm must be of type str, or a subclass of Algorithm.",
        ):
            builder.add(123)

    def test_add_class_not_subclass_of_algorithm_raises_type_error(self):
        class NotAlgo:
            pass

        builder = PipelineBuilder()
        with pytest.raises(TypeError):
            builder.add(NotAlgo)

    def test_add_valid_algorithm_str(self, config_with_modules, monkeypatch):
        builder = PipelineBuilder()

        monkeypatch.setattr(outlier_reduction, "__all__", ["SpinQuant"])

        class FakeAlgo(Algorithm):
            def forward(self, interface):
                raise NotImplementedError()

        monkeypatch.setattr(outlier_reduction, "SpinQuant", FakeAlgo)

        builder.add("SpinQuant")
        assert builder.pipeline[-1] is FakeAlgo
