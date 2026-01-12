import csv
import json
import os

import pytest
from datasets import Dataset

from GenAIQuant.utils.dataset_utils import DatasetProvider


@pytest.fixture(scope="module")
def provider():
    return DatasetProvider()


@pytest.fixture(scope="module")
def json_path():
    path = "test_data.json"
    data = [{"text": "A", "label": 1}, {"text": "B", "label": 0}]
    with open(path, "w") as f:
        json.dump(data, f)
    yield path
    os.remove(path)


@pytest.fixture(scope="module")
def csv_path():
    path = "test_data.csv"
    data = [["text", "label"], ["A", 1], ["B", 0]]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)
    yield path
    os.remove(path)


def test_load_huggingface_dataset_with_split(provider):
    ds = provider.get("wikitext", split="test")
    assert isinstance(ds, Dataset)
    assert len(ds) > 0


def test_load_huggingface_dataset_with_num_samples_and_shuffle(provider):
    ds = provider.get("wikitext", split="test", num_samples=10, shuffle=True)
    assert isinstance(ds, Dataset)
    assert len(ds) == 10


def test_features_property(provider):
    _ = provider.get("wikitext", split="train", num_samples=2)
    features = provider.features
    assert isinstance(features, dict)
    assert "text" in features


def test_load_local_json_without_split(provider, json_path):
    ds = provider.get(json_path)
    assert isinstance(ds, Dataset)
    assert len(ds) == 2


def test_load_local_csv_without_split(provider, csv_path):
    ds = provider.get(csv_path)
    assert isinstance(ds, Dataset)
    assert len(ds) == 2


def test_load_local_json_with_split(provider, json_path):
    ds = provider.get(json_path, split="train")
    assert isinstance(ds, Dataset)
    assert len(ds) == 2


def test_load_local_csv_with_split_and_sampling(provider, csv_path):
    ds = provider.get(csv_path, split="train", num_samples=1)
    assert isinstance(ds, Dataset)
    assert len(ds) == 1


def test_unsupported_dataset_name(provider):
    with pytest.raises(ValueError):
        provider.get("some_unknown_dataset")


def test_invalid_local_path(provider):
    with pytest.raises(ValueError):
        provider.get("this_file_does_not_exist.json")


def test_invalid_file_type(provider):
    path = "temp.txt"
    with open(path, "w") as f:
        f.write("test")
    try:
        with pytest.raises(ValueError):
            provider.get(path)
    finally:
        os.remove(path)


def test_sampling_on_datasetdict_should_fail(provider):
    with pytest.raises(TypeError):
        provider.get("wikitext", split=None, num_samples=10)


def test_features_without_get_should_raise():
    new_provider = DatasetProvider()
    with pytest.raises(RuntimeError):
        _ = new_provider.features
