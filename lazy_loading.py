import re
import torch
from gguf import MODEL_ARCH_NAMES, get_tensor_name_map
from gguf_gpu import dequantize

# Globals to store GGUF state.
GLOBAL_GGUF_MAPPING = {}
GLOBAL_GGUF_READER = None

def get_gguf_hf_weights_map(hf_model, model_type=None, num_layers=None, qual_name=""):
    model_type = hf_model.config.model_type if model_type is None else model_type
    num_layers = hf_model.config.num_hidden_layers if num_layers is None else num_layers
    if model_type == "cohere":
        model_type = "command-r"
    if model_type == "qwen2_moe":
        model_type = "qwen2moe"
    if model_type == "deepseek_v3":
        model_type = "deepseek2"
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
        if model_type == "qwen2moe" and "mlp.experts." in hf_name:
            hf_name = re.sub(r"mlp.experts.\d+.", "mlp.experts.", hf_name)
        name, suffix = hf_name, ""
        if hf_name.endswith(".weight") or hf_name.endswith(".bias"):
            name, suffix = hf_name.rsplit(".", 1)
            suffix = "." + suffix
        name = "model." + name  # required for name map lookup
        gguf_name = name_map.get_name(name)
        if gguf_name is None:
            continue
        gguf_to_hf_name_map[gguf_name + suffix] = qual_name + hf_name
    return gguf_to_hf_name_map

def lazy_load_hook(module, inputs):
    input0 = inputs[0]
    if isinstance(input0, (tuple, list)):
        input0 = input0[0]
    device = input0.device

    for attr, hf_key in getattr(module, "lazy_params", {}).items():
        hf_key = hf_key.replace("model.", "")
        param = getattr(module, attr)
        if param is None or (hasattr(param, "device") and param.device.type == "meta"):
            if "experts" in hf_key and "shared_experts" not in hf_key:
                hf_key = re.sub(r"mlp.experts.\d+.", "mlp.shared_experts.", hf_key)
            if hf_key.endswith("e_score_correction_bias"):
                param_tensor = torch.empty((module.n_routed_experts))
            elif hf_key not in GLOBAL_GGUF_MAPPING:
                raise ValueError(f"GGUF mapping does not contain key: {hf_key}")
            else:
                gguf_tensor = GLOBAL_GGUF_MAPPING[hf_key]
                param_tensor = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
            setattr(module, attr, param_tensor.to(device))

def lazy_offload_hook(module, inputs, output):
    for attr in getattr(module, "lazy_params", {}):
        setattr(module, attr, None)

def get_module_by_name(model, full_param_name):
    parts = full_param_name.split(".")
    mod = model
    for part in parts[:-1]:
        mod = getattr(mod, part)
    return mod

def remove_registered_parameters(model):
    # Do not remove parameters from these modules.
    skip_modules = {"embed_tokens", "rotary_emb", "norm"}
    for full_name, _ in list(model.named_parameters()):
        if full_name.split(".")[0] in skip_modules:
            continue
        if full_name.split(".")[0] == 'layers' and int(full_name.split(".")[1]) < 3:
            # Skip the first 3 Dense layers
            continue
        module = get_module_by_name(model, full_name)
        attr = full_name.split(".")[-1]
        if not hasattr(module, "lazy_params"):
            module.lazy_params = {}
        module.lazy_params[attr] = full_name
        if attr in module._parameters:
            del module._parameters[attr]
        setattr(module, attr, None)

def load_eager_module_weights(module, full_prefix, device="cuda"):
    for full_name, _ in module.named_parameters(recurse=True):
        key = f"{full_prefix}.{full_name}"
        if key not in GLOBAL_GGUF_MAPPING:
            raise ValueError(f"GGUF mapping does not contain key: {key}")
        
        gguf_tensor = GLOBAL_GGUF_MAPPING[key]
        loaded_tensor = dequantize(gguf_tensor.data, gguf_tensor.tensor_type).to(device)
        
        # Split the full_name into its components (e.g. "submodule.weight" -> ["submodule", "weight"])
        name_parts = full_name.split(".")
        # Traverse the module hierarchy to get to the correct submodule
        submodule = module
        for part in name_parts[:-1]:
            submodule = getattr(submodule, part)
        # The actual parameter name (without dots)
        param_name = name_parts[-1]
        # Replace (or register) the parameter in the found submodule
        submodule.register_parameter(param_name, torch.nn.Parameter(loaded_tensor))
