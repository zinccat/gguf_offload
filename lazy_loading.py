import re
import torch
from typing import Optional, Tuple
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
        if model_type in ["qwen2moe", "deepseek2"] and "mlp.experts." in hf_name:
            hf_name = re.sub(r"mlp.experts.\d+.", "mlp.experts.", hf_name)
        if "e_score_correction_bias" in hf_name:
            hf_name = hf_name.replace(
                "e_score_correction_bias", "e_score_correction.bias"
            )
        name, suffix = hf_name, ""
        if hf_name.endswith(".weight") or hf_name.endswith(".bias"):
            name, suffix = hf_name.rsplit(".", 1)
            suffix = "." + suffix
        name = "model." + name  # required for name map lookup
        gguf_name = name_map.get_name(name)
        if gguf_name is None:
            # print(f"Skipping {name} -> {hf_name}")
            continue
        gguf_to_hf_name_map[gguf_name + suffix] = qual_name + hf_name
    return gguf_to_hf_name_map


def lazy_load_hook(module, inputs):
    input0 = inputs[0]
    if isinstance(input0, (tuple, list)):
        input0 = input0[0]

    for attr, hf_key in getattr(module, "lazy_params", {}).items():
        if getattr(module, attr) is not None:
            continue
        expert_idx = None
        param = getattr(module, attr)
        if param is None or (hasattr(param, "device") and param.device.type == "meta"):
            if "experts" in hf_key and "shared_experts" not in hf_key:
                # Use regex to match the expert index part of the key
                match = re.search(r"mlp.experts.(\d+)", hf_key)
                if match:
                    expert_idx = int(match.group(1))  # Get the expert index
                    hf_key = re.sub(
                        r"mlp.experts.(\d+)", "mlp.experts", hf_key
                    )  # Normalize the key
                else:
                    expert_idx = None  # If no expert index is present, set to None
            hf_key = hf_key.replace(
                "e_score_correction_bias", "e_score_correction.bias"
            )
            if hf_key not in GLOBAL_GGUF_MAPPING:
                raise ValueError(f"GGUF mapping does not contain key: {hf_key}")
            else:
                gguf_tensor, dtype = GLOBAL_GGUF_MAPPING[hf_key]
            # print(f"Loaded {hf_key} into {attr}")
            if "weight" not in hf_key or "norm" in hf_key or "gate.weight" in hf_key:
                # all fp32 weights
                param_tensor = gguf_tensor.to("cuda", non_blocking=True)
                if "norm" in hf_key:
                    param_tensor = param_tensor.half()
                setattr(module, attr, param_tensor)
            else:
                if expert_idx is not None:
                    setattr(
                        module,
                        attr,
                        gguf_tensor[expert_idx].to("cuda", non_blocking=True),
                    )
                else:
                    setattr(module, attr, gguf_tensor.to("cuda", non_blocking=True))
                setattr(module, "weight_type", int(dtype))


def load_experts(experts, layer_idx, experts_idx):
    keys = ["up_proj.weight", "down_proj.weight", "gate_proj.weight"]
    for key in keys:
        hf_key = f"layers.{layer_idx}.mlp.experts.{key}"
        if hf_key not in GLOBAL_GGUF_MAPPING:
            raise ValueError(f"GGUF mapping does not contain key: {hf_key}")
        gguf_tensor, dtype = GLOBAL_GGUF_MAPPING[hf_key]
        # param_tensor = dequantize(
        #     gguf_tensor.data, gguf_tensor.tensor_type, experts_idx
        # )
        # setattr(experts, key, param_tensor)
        for i, idx in enumerate(experts_idx):
            setattr(experts[idx], key, gguf_tensor[i].to("cuda", non_blocking=True))
            setattr(experts[idx], "weight_type", int(dtype))


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
        module = get_module_by_name(model, full_name)
        if full_name.split(".")[0] in skip_modules:
            # setattr(module, "load_once", True)
            continue
        if full_name.split(".")[0] == "layers" and int(full_name.split(".")[1]) < 3:
            # Skip the first 3 Dense layers
            setattr(module, "load_once", True)
        if (
            "shared_experts" in full_name
            or "mlp.gate" in full_name
            or "norm" in full_name
            or "self_attn" in full_name
        ):
            setattr(module, "load_once", True)
        elif "experts" not in full_name:
            print(f"Lazy loading {full_name}")
        attr = full_name.split(".")[-1]
        if not hasattr(module, "lazy_params"):
            module.lazy_params = {}
        module.lazy_params[attr] = full_name.replace("model.", "")
        if attr in module._parameters:
            del module._parameters[attr]
        setattr(module, attr, None)


def load_eager_module_weights(module, full_prefix, device="cuda"):
    for full_name, _ in module.named_parameters(recurse=True):
        if "experts" in full_name and "shared_experts" not in full_name:
            full_name = re.sub(r"mlp.experts.\d+.", "mlp.experts.", full_name)
        if full_name.endswith("e_score_correction_bias"):
            full_name = full_name.replace(
                "e_score_correction_bias", "e_score_correction.bias"
            )
        else:
            key = f"{full_prefix}.{full_name}"
            if key not in GLOBAL_GGUF_MAPPING:
                raise ValueError(f"GGUF mapping does not contain key: {key}")

            gguf_tensor, dtype = GLOBAL_GGUF_MAPPING[key]
            # loaded_tensor = dequantize(gguf_tensor.data, gguf_tensor.tensor_type).to(
            #     device, non_blocking=True
            # )
            loaded_tensor = dequantize(gguf_tensor, dtype).to(device, non_blocking=True)
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


def pipelined_inference_layers(
    layers,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    y = hidden_states

    for layer in layers:
        layer_output = layer(
            hidden_states=y,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        y = layer_output[0]

    if use_cache:
        return y, layer_output[1]

    # torch.cuda.empty_cache()
    return y
