"""
Adapted from https://github.com/99991/pygguf and Transformers
"""

import re
from typing import Dict, NamedTuple, Optional

import torch
import numpy as np
from tqdm import tqdm
import inspect
from transformers.utils.logging import get_logger
from transformers.integrations import (
    GGUF_CONFIG_MAPPING,
    GGUF_TOKENIZER_MAPPING,
    _gguf_parse_value,
)
from transformers import PreTrainedModel, PretrainedConfig
from gguf import MODEL_ARCH_NAMES, get_tensor_name_map, GGUFReader, dequantize
from accelerate.utils import (
        offload_weight,
        set_module_tensor_to_device
    )


logger = get_logger(__name__)

GGUF_TO_TRANSFORMERS_MAPPING = {
    "ignore": {
        "GGUF": {
            "version": "version",
            "tensor_count": "tensor_count",
            "kv_count": "kv_count",
        },
        "general": {"file_type": "file_type", "quantization_version": "quantization_version"},
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

class Qwen2MoeTensorProcessor(TensorProcessor):
    def __init__(self, config=None):
        super().__init__(config=config)

    def process(self, weights, name, **kwargs):
        if "_exp" in name:
            tensor_key_mapping = kwargs.get("tensor_key_mapping")
            parsed_parameters = kwargs.get("parsed_parameters")
            if tensor_key_mapping:
                self._split_moe_expert_tensor(weights, parsed_parameters, name, tensor_key_mapping)
                return GGUFTensor(weights, None, {})
        if "ffn_gate_inp_shexp" in name:
            # for compatibility tensor shared_expert_gate must be (1, 2048) dim,
            # quantized one is (2048)
            weights = np.expand_dims(weights, axis=0)
        return GGUFTensor(weights, name, {})

    def _split_moe_expert_tensor(
        self, weights: np.ndarray, parsed_parameters: Dict[str, Dict], name: str, tensor_key_mapping: dict
    ):
        # Original merge implementation
        # https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py#L1994-L2022
        name = tensor_key_mapping[name]
        w_counter = self.config.get("num_experts", 60)
        for i in range(0, w_counter):
            temp_name = name.replace("mlp.experts.", f"mlp.experts.{i}.")
            exp_weight = weights[i]
            parsed_parameters["tensors"][temp_name] = torch.from_numpy(np.copy(exp_weight))


TENSOR_PROCESSORS = {
    # "llama": LlamaTensorProcessor,
    "qwen2moe": Qwen2MoeTensorProcessor,
    # "bloom": BloomTensorProcessor,
    # "t5": T5TensorProcessor,
    # "t5encoder": T5TensorProcessor,
    # "gpt2": GPT2TensorProcessor,
    # "mamba": MambaTensorProcessor,
    # "gemma2": Gemma2TensorProcessor,
}

def read_field(reader, field):
    value = reader.fields[field]
    return [_gguf_parse_value(value.parts[_data_index], value.types) for _data_index in value.data]

def get_gguf_hf_weights_map(
    hf_model,
    model_type: Optional[str] = None,
    num_layers: Optional[int] = None,
    qual_name: str = "",
):
    """
    GGUF uses this naming convention for their tensors from HF checkpoint:
    `blk.N.BB.weight` and `blk.N.BB.bias`
    where N signifies the block number of a layer, and BB signifies the
    attention/mlp layer components.
    See "Standardized tensor names" in
    https://github.com/ggerganov/ggml/blob/master/docs/gguf.md for details.
    """
    print(hf_model.config)
    model_type = hf_model.config.model_type if model_type is None else model_type
    num_layers = hf_model.config.num_hidden_layers if num_layers is None else num_layers
    # hack: ggufs have a different name for cohere
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
        raise NotImplementedError(
            f"Unknown gguf model_type: {model_type} in gguf-py. "
            "This might because you're using an outdated version of gguf-py package, "
            "you can install `gguf` package from source refer to "
            "https://github.com/ggerganov/llama.cpp/tree/master/gguf-py#development"
        )
    name_map = get_tensor_name_map(arch, num_layers)

    # Use a dummy conversion to get the mapping, because
    # hf => gguf and gguf => hf mappings are reversed
    gguf_to_hf_name_map = {}
    state_dict = hf_model.state_dict()
    for hf_name in state_dict.keys():
        # An exception for qwen2moe model, where the expert layers are packed
        if model_type == "qwen2moe" and "mlp.experts." in hf_name:
            hf_name = re.sub(r"mlp.experts.\d+.", "mlp.experts.", hf_name)

        name, suffix = hf_name, ""
        if hf_name.endswith(".weight") or hf_name.endswith(".bias"):
            name, suffix = hf_name.rsplit(".", 1)
            suffix = "." + suffix

        gguf_name = name_map.get_name(name)
        if gguf_name is None:
            continue

        gguf_to_hf_name_map[gguf_name + suffix] = qual_name + hf_name

    # Some model like Bloom converted from BloomModel instead of BloomForCausalLM
    # Therefore, we need to check submodule as well to get a correct mapping
    if named_children := hf_model.named_children():
        for name, child in named_children:
            sub_map = get_gguf_hf_weights_map(child, model_type, num_layers, qual_name=f"{qual_name}{name}.")
            # Ignore the keys that are already in the main map to avoid overwriting
            sub_map = {k: v for k, v in sub_map.items() if k not in gguf_to_hf_name_map}
            gguf_to_hf_name_map.update(sub_map)

    return gguf_to_hf_name_map

def load_gguf_checkpoint(gguf_checkpoint_path, return_tensors=False, model_to_load=None):
    """
    Load a GGUF file and return a dictionary of parsed parameters containing tensors, the parsed
    tokenizer and config attributes.

    Args:
        gguf_checkpoint_path (`str`):
            The path the to GGUF file to load
        return_tensors (`bool`, defaults to `True`):
            Whether to read the tensors from the file and return them. Not doing so is faster
            and only loads the metadata in memory.
    """
    reader = GGUFReader(gguf_checkpoint_path)
    fields = reader.fields
    reader_keys = list(fields.keys())

    parsed_parameters = {k: {} for k in GGUF_TO_TRANSFORMERS_MAPPING}

    architecture = read_field(reader, "general.architecture")[0]
    model_name = read_field(reader, "general.name")

    # in llama.cpp mistral models use the same architecture as llama. We need
    # to add this patch to ensure things work correctly on our side.
    if "llama" in architecture and "mistral" in model_name:
        updated_architecture = "mistral"
    # FIXME: Currnetly this implementation is only for flan-t5 architecture.
    # It needs to be developed for supporting legacy t5.
    elif "t5" in architecture or "t5encoder" in architecture:
        parsed_parameters["config"]["is_gated_act"] = True
        updated_architecture = "t5"
    else:
        updated_architecture = architecture

    if "qwen2moe" in architecture:
        updated_architecture = "qwen2_moe"

    # For stablelm architecture, we need to set qkv_bias and use_parallel_residual from tensors
    # If `qkv_bias=True`, qkv_proj with bias will be present in the tensors
    # If `use_parallel_residual=False`, ffn_norm will be present in the tensors
    if "stablelm" in architecture:
        attn_bias_name = {"attn_q.bias", "attn_k.bias", "attn_v.bias"}
        ffn_norm_name = "ffn_norm"
        qkv_bias = any(bias_name in tensor.name for tensor in reader.tensors for bias_name in attn_bias_name)
        use_parallel_residual = any(ffn_norm_name in tensor.name for tensor in reader.tensors)
        parsed_parameters["config"]["use_qkv_bias"] = qkv_bias
        parsed_parameters["config"]["use_parallel_residual"] = not use_parallel_residual

    if architecture not in GGUF_SUPPORTED_ARCHITECTURES:
        raise ValueError(f"GGUF model with architecture {architecture} is not supported yet.")

    # Handle tie_word_embeddings, if lm_head.weight is not present in tensors,
    # tie_word_embeddings is true otherwise false
    parsed_parameters["config"]["tie_word_embeddings"] = all(
        "output.weight" != tensor.name for tensor in reader.tensors
    )

    # List all key-value pairs in a columnized format
    for gguf_key, field in reader.fields.items():
        gguf_key = gguf_key.replace(architecture, updated_architecture)
        split = gguf_key.split(".")
        prefix = split[0]
        config_key = ".".join(split[1:])

        value = [_gguf_parse_value(field.parts[_data_index], field.types) for _data_index in field.data]

        if len(value) == 1:
            value = value[0]

        if isinstance(value, str) and architecture in value:
            value = value.replace(architecture, updated_architecture)

        for parameter in GGUF_TO_TRANSFORMERS_MAPPING:
            parameter_renames = GGUF_TO_TRANSFORMERS_MAPPING[parameter]
            if prefix in parameter_renames and config_key in parameter_renames[prefix]:
                renamed_config_key = parameter_renames[prefix][config_key]
                if renamed_config_key == -1:
                    continue

                if renamed_config_key is not None:
                    parsed_parameters[parameter][renamed_config_key] = value

                if gguf_key in reader_keys:
                    reader_keys.remove(gguf_key)

        if gguf_key in reader_keys:
            logger.info(f"Some keys were not parsed and added into account {gguf_key} | {value}")

    # retrieve config vocab_size from tokenizer
    # Pleas refer to https://github.com/huggingface/transformers/issues/32526 for more details
    if "vocab_size" not in parsed_parameters["config"]:
        tokenizer_parameters = parsed_parameters["tokenizer"]
        if "tokens" in tokenizer_parameters:
            parsed_parameters["config"]["vocab_size"] = len(tokenizer_parameters["tokens"])
        else:
            logger.warning(
                "Can't find a way to retrieve missing config vocab_size from tokenizer parameters. "
                "This will use default value from model config class and cause unexpected behavior."
            )

    if return_tensors:
        parsed_parameters["tensors"] = {}

        tensor_key_mapping = get_gguf_hf_weights_map(model_to_load)
        config = parsed_parameters.get("config", {})

        ProcessorClass = TENSOR_PROCESSORS.get(architecture, TensorProcessor)
        processor = ProcessorClass(config=config)
        for tensor in tqdm(reader.tensors, desc="Converting and de-quantizing GGUF tensors..."):
            name = tensor.name
            weights = dequantize(tensor.data, tensor.tensor_type)

            result = processor.process(
                weights=weights,
                name=name,
                tensor_key_mapping=tensor_key_mapping,
                parsed_parameters=parsed_parameters,
            )

            weights = result.weights
            name = result.name

            if name not in tensor_key_mapping:
                continue

            name = tensor_key_mapping[name]

            parsed_parameters["tensors"][name] = torch.from_numpy(np.copy(weights))

    if len(reader_keys) > 0:
        logger.info(f"Some keys of the GGUF file were not considered: {reader_keys}")

    return parsed_parameters

def _load_state_dict_into_meta_model(
    model,
    state_dict,
    start_prefix,
    expected_keys,
    device_map=None,
    offload_folder=None,
    offload_index=None,
    state_dict_folder=None,
    state_dict_index=None,
    dtype=None,
    hf_quantizer=None,
    is_safetensors=False,
    keep_in_fp32_modules=None,
    unexpected_keys=None,  # passing `unexpected` for cleanup from quantization items
    pretrained_model_name_or_path=None,  # for flagging the user when the model contains renamed keys
):
    """
    This is somewhat similar to `_load_state_dict_into_model`, but deals with a model that has some or all of its
    params on a `meta` device. It replaces the model params with the data from the `state_dict`, while moving the
    params back to the normal device, but only for `loaded_state_dict_keys`.

    `start_prefix` is used for models which insert their name into model keys, e.g. `bert` in
    `bert.pooler.dense.weight`

    """

    # XXX: remaining features to implement to be fully compatible with _load_state_dict_into_model
    # - deepspeed zero 3 support
    # - need to copy metadata if any - see _load_state_dict_into_model
    # - handling error_msgs - mimicking the error handling in module._load_from_state_dict()

    error_msgs = []

    is_quantized = hf_quantizer is not None

    is_torch_e4m3fn_available = hasattr(torch, "float8_e4m3fn")

    for param_name, param in state_dict.items():
        if param_name not in expected_keys:
            continue

        if param_name.startswith(start_prefix):
            param_name = param_name[len(start_prefix) :]

        module_name = param_name
        set_module_kwargs = {}

        # We convert floating dtypes to the `dtype` passed except for float8_e4m3fn type. We also want to keep the buffers/params
        # in int/uint/bool and not cast them.
        is_param_float8_e4m3fn = is_torch_e4m3fn_available and param.dtype == torch.float8_e4m3fn
        if dtype is not None and torch.is_floating_point(param) and not is_param_float8_e4m3fn:
            if (
                keep_in_fp32_modules is not None
                and any(
                    module_to_keep_in_fp32 in param_name.split(".") for module_to_keep_in_fp32 in keep_in_fp32_modules
                )
                and dtype == torch.float16
            ):
                param = param.to(torch.float32)

                # For backward compatibility with older versions of `accelerate`
                # TODO: @sgugger replace this check with version check at the next `accelerate` release
                if "dtype" in list(inspect.signature(set_module_tensor_to_device).parameters):
                    set_module_kwargs["dtype"] = torch.float32
            else:
                param = param.to(dtype)

        # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model, and which
        # uses `param.copy_(input_param)` that preserves the contiguity of the parameter in the model.
        # Reference: https://github.com/pytorch/pytorch/blob/db79ceb110f6646523019a59bbd7b838f43d4a86/torch/nn/modules/module.py#L2040C29-L2040C29

        old_param = model
        splits = param_name.split(".")
        for split in splits:
            # We shouldn't hit the default value unless for quant methods like hqq that modifies expected_keys.
            old_param = getattr(old_param, split, None)
            if old_param is None:
                break

        if not isinstance(old_param, (torch.nn.Parameter, torch.Tensor)):
            old_param = None

        if old_param is not None:
            if dtype is None:
                param = param.to(old_param.dtype)

            if old_param.is_contiguous():
                param = param.contiguous()

        set_module_kwargs["value"] = param

        if device_map is None:
            param_device = "cpu"
        else:
            # find next higher level module that is defined in device_map:
            # bert.lm_head.weight -> bert.lm_head -> bert -> ''
            while len(module_name) > 0 and module_name not in device_map:
                module_name = ".".join(module_name.split(".")[:-1])
            if module_name == "" and "" not in device_map:
                # TODO: group all errors and raise at the end.
                raise ValueError(f"{param_name} doesn't have any device set.")
            param_device = device_map[module_name]

        if param_device == "disk":
            if not is_safetensors:
                offload_index = offload_weight(param, param_name, offload_folder, offload_index)
        elif param_device == "cpu" and state_dict_index is not None:
            state_dict_index = offload_weight(param, param_name, state_dict_folder, state_dict_index)
        elif (
            not is_quantized
            or (not hf_quantizer.requires_parameters_quantization)
            or (
                not hf_quantizer.check_quantized_param(
                    model, param, param_name, state_dict, param_device=param_device, device_map=device_map
                )
            )
        ):
            # For backward compatibility with older versions of `accelerate` and for non-quantized params
            set_module_tensor_to_device(model, param_name, param_device, **set_module_kwargs)
        else:
            hf_quantizer.create_quantized_param(model, param, param_name, param_device, state_dict, unexpected_keys)

    return error_msgs, offload_index, state_dict_index


if __name__ == "__main__":
    # model = load_gguf_checkpoint("/home/zinccat/codes/OffloadTest/gg/qwen2.5-coder-0.5b-instruct-q4_0.gguf")
    config = PretrainedConfig.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
    gguf_path = "/home/zinccat/codes/OffloadTest/gg/qwen2.5-coder-0.5b-instruct-q4_0.gguf"
    with torch.device("meta"):
        dummy_model = PreTrainedModel(config)
    # benchmark dequantization
    from timeit import default_timer as timer
    start = timer()
    for _ in range(1):
        state_dict = load_gguf_checkpoint(gguf_path, return_tensors=True, model_to_load=dummy_model)["tensors"]
    # state_dict = load_gguf_checkpoint(gguf_path, return_tensors=True, model_to_load=dummy_model)["tensors"]
    
    print("Time elapsed:", timer() - start)

    error_msgs, offload_index, state_dict_index = _load_state_dict_into_meta_model(
        dummy_model,
        state_dict,
        "",
        state_dict.keys(),
        device_map=None,
        offload_folder=None,
        offload_index=None,
        state_dict_folder=None,
        state_dict_index=None,
        dtype=None,
        hf_quantizer=None,
        is_safetensors=False,
        keep_in_fp32_modules=None,
        unexpected_keys=None,
        pretrained_model_name_or_path=None,
    )
    print(error_msgs)
    print(dummy_model)