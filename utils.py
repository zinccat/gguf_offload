from typing import NamedTuple

import numpy as np
from transformers.integrations import (
    GGUF_CONFIG_MAPPING,
    GGUF_TOKENIZER_MAPPING,
)

# === GGUF and Tensor Processing Setup ===

GGUF_TO_TRANSFORMERS_MAPPING = {
    "ignore": {
        "GGUF": {
            "version": "version",
            "tensor_count": "tensor_count",
            "kv_count": "kv_count",
        },
        "general": {
            "file_type": "file_type",
            "quantization_version": "quantization_version",
        },
    },
    "config": GGUF_CONFIG_MAPPING,
    "tokenizer": {"tokenizer": GGUF_TOKENIZER_MAPPING["tokenizer"]},
    "tokenizer_config": {"tokenizer": GGUF_TOKENIZER_MAPPING["tokenizer_config"]},
}

GGUF_SUPPORTED_ARCHITECTURES = list(GGUF_TO_TRANSFORMERS_MAPPING["config"].keys())


class GGUFTensor(NamedTuple):
    weights: np.ndarray
    name: str
    metadata: dict


class TensorProcessor:
    def __init__(self, config=None):
        self.config = config or {}

    def process(self, weights, name, **kwargs):
        return GGUFTensor(weights, name, {})


TENSOR_PROCESSORS = {}
