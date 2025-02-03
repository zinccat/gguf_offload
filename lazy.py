"""
Adapted from https://github.com/99991/pygguf and Transformers,
with lazy per-layer dequantization/loading of weights for the Qwen model.
"""

import re
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

from transformers.utils import ContextManagers
from transformers.utils.logging import get_logger
from transformers import PretrainedConfig, Qwen2Model
from transformers.modeling_utils import no_init_weights, init_empty_weights

from gguf import MODEL_ARCH_NAMES, get_tensor_name_map, GGUFReader, dequantize

# Set random seed for reproducibility
torch.manual_seed(0)
logger = get_logger(__name__)

def get_gguf_hf_weights_map(
    hf_model,
    model_type: Optional[str] = None,
    num_layers: Optional[int] = None,
    qual_name: str = "",
):
    """
    Creates a mapping from GGUF tensor names to HF parameter names.
    """
    model_type = hf_model.config.model_type if model_type is None else model_type
    num_layers = hf_model.config.num_hidden_layers if num_layers is None else num_layers
    if model_type == "cohere":
        model_type = "command-r"
    if model_type == "qwen2_moe":
        model_type = "qwen2moe"
    arch = None
    for key, value in MODEL_ARCH_NAMES.items():
        if value == model_type:
            arch = key
            break
    if arch is None:
        raise NotImplementedError(f"Unknown gguf model_type: {model_type}")
    name_map = get_tensor_name_map(arch, num_layers)
    gguf_to_hf_name_map = {}
    state_dict = hf_model.state_dict()
    for hf_name in state_dict.keys():
        # For qwen2moe, normalize expert names.
        if model_type == "qwen2moe" and "mlp.experts." in hf_name:
            hf_name = re.sub(r"mlp.experts.\d+.", "mlp.experts.", hf_name)
        name, suffix = hf_name, ""
        if hf_name.endswith(".weight") or hf_name.endswith(".bias"):
            name, suffix = hf_name.rsplit(".", 1)
            suffix = "." + suffix
        name = "model." + name  # essential for the name_map.get_name to work
        gguf_name = name_map.get_name(name)
        if gguf_name is None:
            continue
        gguf_to_hf_name_map[gguf_name + suffix] = qual_name + hf_name
    return gguf_to_hf_name_map


# === Lazy-Loading / Offloading Setup for Individual Layers ===

# Instead of dequantizing all weights at once,
# we build a global mapping from each HF parameter name to its corresponding GGUF tensor object.
GLOBAL_GGUF_READER = None
GLOBAL_GGUF_MAPPING = {}


def lazy_load_hook(module, inputs):
    """
    Pre-forward hook:
    For each lazy parameter in the module, look up its GGUF tensor,
    dequantize it on demand, and move it to the device of the input.
    """
    for attr, hf_key in getattr(module, "lazy_params", {}).items():
        if getattr(module, attr) is None:
            if hf_key not in GLOBAL_GGUF_MAPPING:
                raise ValueError(f"GGUF mapping does not contain key: {hf_key}")
            gguf_tensor = GLOBAL_GGUF_MAPPING[hf_key]
            # Dequantize the weight individually on demand.
            deq_weights = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
            # Convert to torch tensor.
            param_tensor = torch.from_numpy(np.copy(deq_weights))
            setattr(module, attr, param_tensor.to(inputs[0].device))


def lazy_offload_hook(module, inputs, output):
    """
    Post-forward hook:
    After computation, offload the parameter by setting it back to None.
    """
    for attr in getattr(module, "lazy_params", {}):
        setattr(module, attr, None)


def get_module_by_name(model: nn.Module, full_param_name: str) -> nn.Module:
    """Return the parent module that holds the parameter given its full name."""
    parts = full_param_name.split(".")
    mod = model
    for part in parts[:-1]:
        mod = getattr(mod, part)
    return mod


def remove_registered_parameters(model: nn.Module):
    """
    For every registered parameter in the model, remove it from the module
    (so it does not occupy memory) and record its full key in a new attribute `lazy_params`.
    """
    for full_name, param in list(model.named_parameters()):
        module = get_module_by_name(model, full_name)
        attr = full_name.split(".")[-1]
        if not hasattr(module, "lazy_params"):
            module.lazy_params = {}
        # Record the mapping from attribute (e.g. 'weight') to the full parameter name (HF key)
        module.lazy_params[attr] = full_name
        # Remove the parameter from the module’s _parameters and set attribute to None.
        if attr in module._parameters:
            del module._parameters[attr]
        setattr(module, attr, None)


# === Main Loading and Model Setup ===

if __name__ == "__main__":
    # Specify the pretrained model name/path and GGUF checkpoint.
    pretrained_model_name_or_path = "Qwen/Qwen2.5-Coder-0.5B"
    config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path)
    gguf_path = "models/qwen2.5-coder-0.5b-instruct-q4_0.gguf"

    # Create a dummy model on the 'meta' device to build the tensor mapping.
    with torch.device("meta"):
        dummy_model = Qwen2Model(config)
    # Build the mapping from GGUF tensor names to HF parameter names.
    tensor_key_mapping = get_gguf_hf_weights_map(dummy_model)

    # Initialize the GGUF reader (this loads only metadata and tensor pointers).
    GLOBAL_GGUF_READER = GGUFReader(gguf_path)
    # Build GLOBAL_GGUF_MAPPING: map each HF parameter name to its corresponding GGUF tensor.
    for tensor in GLOBAL_GGUF_READER.tensors:
        name = tensor.name
        if name not in tensor_key_mapping:
            continue
        hf_key = tensor_key_mapping[name]
        GLOBAL_GGUF_MAPPING[hf_key] = tensor

    # Initialize the actual model on empty weights.
    _fast_init = True
    init_contexts = [no_init_weights(_enable=_fast_init), init_empty_weights()]
    with ContextManagers(init_contexts):
        model = Qwen2Model(config)

    # Remove registered parameters so that each layer will load its weight on demand.
    remove_registered_parameters(model)

    # Attach the lazy loading/offloading hooks to every module that has lazy parameters.
    for module in model.modules():
        if hasattr(module, "lazy_params"):
            module.register_forward_pre_hook(lazy_load_hook)
            module.register_forward_hook(lazy_offload_hook)

    model.eval()

    with torch.no_grad():
        # Adjust input dimensions as required by your model; here we use 512 tokens as an example.
        input_ids = torch.randint(0, 151936, (1, 512))
        output = model(input_ids)
        print("Model output:", output)

    from timeit import default_timer as timer

    model.rotary_emb.to("cuda")

    start = timer()

    # Run a forward pass.
    # Each layer will dequantize its weight individually on demand, perform computation, then offload.
    with torch.no_grad():
        for i in range(10):
            # Adjust input dimensions as required by your model; here we use 512 tokens as an example.
            input_ids = torch.randint(0, 151936, (1, 512)).cuda()
            output = model(input_ids)

    end = timer()
    print("Time taken:", (end - start) / 10)
