"""
Running dequantization lazily during inference with GGUF using GPU for DeepSeek-R1 model.

Partly adapted from https://github.com/99991/pygguf and Transformers
"""

import re
import torch
import torch.nn as nn

from transformers import PretrainedConfig
from transformers.modeling_utils import no_init_weights, init_empty_weights
from transformers.utils import ContextManagers
from transformers.utils.logging import get_logger
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.utils.import_utils import is_torch_fx_available

from gguf import MODEL_ARCH_NAMES, get_tensor_name_map, GGUFReader
from gguf_gpu import dequantize

from deepseek.modeling_deepseek import DeepseekV3Model

torch.manual_seed(0)
logger = get_logger(__name__)

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

if is_torch_fx_available():
    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


# --- Lazy Loading Utilities (unchanged except for use in meta tensors) ---
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


GLOBAL_GGUF_READER = None
GLOBAL_GGUF_MAPPING = {}


def lazy_load_hook(module, inputs):
    # Determine the device from the input tensor.
    input0 = inputs[0]
    if isinstance(input0, tuple) or isinstance(input0, list):
        # If the first input is a tuple/list, take its first element.
        input0 = input0[0]
    device = input0.device

    for attr, hf_key in getattr(module, "lazy_params", {}).items():
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
            # Use the device determined above.
            setattr(module, attr, param_tensor.to(device))


def lazy_offload_hook(module, inputs, output):
    for attr in getattr(module, "lazy_params", {}):
        setattr(module, attr, None)


def get_module_by_name(model: nn.Module, full_param_name: str) -> nn.Module:
    parts = full_param_name.split(".")
    mod = model
    for part in parts[:-1]:
        mod = getattr(mod, part)
    return mod


def remove_registered_parameters(model: nn.Module):
    """
    Remove parameters from the model to enable lazy loading,
    except for the specified modules that we want to keep loaded (embed_tokens, rotary_emb, norm).
    """
    # Specify the top-level modules to keep (do not remove their parameters)
    skip_modules = {"embed_tokens", "rotary_emb", "norm"}
    for full_name, param in list(model.named_parameters()):
        # full_name looks like "embed_tokens.weight", "rotary_emb.some_attr", etc.
        if full_name.split(".")[0] in skip_modules:
            # Skip these so their parameters remain intact.
            continue
        module = get_module_by_name(model, full_name)
        attr = full_name.split(".")[-1]
        if not hasattr(module, "lazy_params"):
            module.lazy_params = {}
        module.lazy_params[attr] = full_name
        if attr in module._parameters:
            del module._parameters[attr]
        setattr(module, attr, None)


# --- Pipelined Inference Utilities for Decoder Layers ---
def chunked_layers(layers, chunk_size=4):
    layers_list = list(layers)
    return [
        layers_list[i : i + chunk_size] for i in range(0, len(layers_list), chunk_size)
    ]


def clone_module(module, memo=None):
    if memo is None:
        memo = {}
    if not isinstance(module, nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    # Explicitly copy lazy_params if it exists.
    # if hasattr(module, "lazy_params"):
    #     # Make a shallow copy (or deep copy if necessary)
    #     clone.lazy_params = module.lazy_params.copy()
    return clone


def copy_layers_to_device(layers, device, non_blocking=True):
    new_layers = []
    for layer in layers:
        new_layer = clone_module(layer)
        if hasattr(new_layer, "lazy_params"):
            new_layer.register_forward_pre_hook(lazy_load_hook)
            new_layer.register_forward_hook(lazy_offload_hook)
        new_layers.append(new_layer.to(device, non_blocking=non_blocking))
    return new_layers


def run_chunk(chunk, x, **kwargs):
    """
    Run a forward pass through a chunk (list) of layers.
    All extra keyword arguments (such as attention_mask, position_ids,
    past_key_value, cache_position, position_embeddings, etc.) are forwarded.
    """
    with torch.no_grad():
        for layer in chunk:
            x = layer(x, **kwargs)[0]
    return x


def pipelined_inference_layers(layers, x, chunk_size=4, **kwargs):
    """
    Runs pipelined inference on a list of layers.
    Splits layers into chunks, clones each chunk to GPU,
    and processes them in a pipelined fashion while forwarding extra keyword arguments.
    """
    if not torch.cuda.is_available():
        with torch.no_grad():
            for layer in layers:
                x = layer(x, **kwargs)
        return x

    device = torch.device("cuda")
    net_chunks = chunked_layers(layers, chunk_size)
    num_chunks = len(net_chunks)

    load_stream = torch.cuda.Stream()
    comp_stream = torch.cuda.Stream()
    cleanup_stream = torch.cuda.Stream()

    compute_done_events = [
        torch.cuda.Event(enable_timing=False) for _ in range(num_chunks)
    ]
    load_done_events = [None] * num_chunks
    gpu_chunks = [None] * num_chunks

    # 1) Move input to GPU.
    with torch.cuda.stream(comp_stream):
        y = x.to(device, non_blocking=True)

    # 2) Pre-load the first chunk.
    with torch.cuda.stream(load_stream):
        gpu_chunks[0] = copy_layers_to_device(net_chunks[0], device, non_blocking=True)
        load_done_events[0] = torch.cuda.Event(enable_timing=False)
        load_done_events[0].record(load_stream)

    # 3) Process chunks in a pipelined fashion.
    for i in range(num_chunks):
        comp_stream.wait_event(load_done_events[i])
        with torch.cuda.stream(comp_stream):
            y = run_chunk(gpu_chunks[i], y, **kwargs)
            compute_done_events[i].record(comp_stream)
        if i + 1 < num_chunks:
            with torch.cuda.stream(load_stream):
                gpu_chunks[i + 1] = copy_layers_to_device(
                    net_chunks[i + 1], device, non_blocking=True
                )
                load_done_events[i + 1] = torch.cuda.Event(enable_timing=False)
                load_done_events[i + 1].record(load_stream)
        with torch.cuda.stream(cleanup_stream):
            cleanup_stream.wait_event(compute_done_events[i])
            gpu_chunks[i] = None  # Release GPU copy.

    comp_stream.synchronize()
    load_stream.synchronize()
    cleanup_stream.synchronize()

    # torch.cuda.empty_cache()

    return y


# --- Main Model Loading and Inference ---

if __name__ == "__main__":
    # Specify pretrained model and GGUF checkpoint.
    pretrained_model_name_or_path = "deepseek-ai/DeepSeek-R1"
    config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path)

    # Create a dummy model on "meta" to build the tensor mapping.
    with torch.device("meta"):
        dummy_model = DeepseekV3Model(config)
    tensor_key_mapping = get_gguf_hf_weights_map(dummy_model)

    for i in range(1, 4):
        gguf_path = f"../DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-0000{i}-of-00003.gguf"
        # Initialize the GGUF reader (loads metadata and tensor pointers).
        GLOBAL_GGUF_READER = GGUFReader(gguf_path)
        for tensor in GLOBAL_GGUF_READER.tensors:
            name = tensor.name
            if name not in tensor_key_mapping:
                continue
            hf_key = tensor_key_mapping[name]
            GLOBAL_GGUF_MAPPING[hf_key] = tensor

    # Initialize the actual model with empty weights.
    init_contexts = [no_init_weights(_enable=True), init_empty_weights()]
    with ContextManagers(init_contexts):
        model = DeepseekV3Model(config)

    # Remove parameters for lazy loading, except for embed_tokens and norm.
    remove_registered_parameters(model)

    # Attach lazy loading/offloading hooks for modules that have lazy_params.
    for module in model.modules():
        if hasattr(module, "lazy_params"):
            module.register_forward_pre_hook(lazy_load_hook)
            module.register_forward_hook(lazy_offload_hook)

    # Set the model to eval mode.
    model.eval()

    def load_eager_module_weights(module: nn.Module, full_prefix: str, device="cuda"):
        """
        Eagerly load weights for a module whose parameters are still meta.
        full_prefix should be the prefix used in the GGUF mapping (e.g., "model.rotary_emb").
        This function iterates over the module's immediate parameters and loads their weight data.
        """
        for name, param in module.named_parameters(recurse=False):
            key = f"{full_prefix}.{name}"
            if key not in GLOBAL_GGUF_MAPPING:
                raise ValueError(f"GGUF mapping does not contain key: {key}")
            gguf_tensor = GLOBAL_GGUF_MAPPING[key]
            # dequantize returns a numpy array; copy it into a torch tensor.
            loaded_tensor = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
            # Replace the parameter with the loaded tensor wrapped in a Parameter.
            module.register_parameter(name, nn.Parameter(loaded_tensor))

    print(model)

    # For modules we want permanently on GPU:
    load_eager_module_weights(model.embed_tokens, "embed_tokens")
    load_eager_module_weights(model.norm, "norm")

    # Explicitly move the modules we want to remain on GPU.
    model.embed_tokens.to("cuda")
    model.norm.to("cuda")

    # --- Inference Example ---
    with torch.no_grad():
        # Create dummy input.
        batch_size, seq_length = 1, 512
        input_ids = torch.randint(0, 129280, (batch_size, seq_length)).cuda()
        x = model.embed_tokens(input_ids)
        past_key_values = None
        cache_position = torch.arange(0, x.shape[1], device=x.device)
        position_ids = cache_position.unsqueeze(0)
        attention_mask = None
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), x, 0
        )
        extra_kwargs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_value": past_key_values,
            "output_attentions": False,
            "use_cache": False,
            "cache_position": cache_position,
        }
        # Process only the decoder layers using pipelined inference.
        x = pipelined_inference_layers(model.layers, x, chunk_size=12, **extra_kwargs)
        x = model.norm(x)
        print("Final output:", x[0, 0, :5])

    from timeit import default_timer as timer

    start = timer()
    with torch.no_grad():
        for _ in range(5):
            batch_size, seq_length = 1, 512
            input_ids = torch.randint(0, 129280, (batch_size, seq_length)).cuda()
            x = model.embed_tokens(input_ids)
            past_key_values = None
            cache_position = torch.arange(0, x.shape[1], device=x.device)
            position_ids = cache_position.unsqueeze(0)
            attention_mask = None
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), x, 0
            )
            extra_kwargs = {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_value": past_key_values,
                "output_attentions": False,
                "use_cache": False,
                "cache_position": cache_position,
            }
            x = pipelined_inference_layers(
                model.layers, x, chunk_size=12, **extra_kwargs
            )
            x = model.norm(x)
            print(f"Finished {_ + 1} inference, used {torch.cuda.memory_allocated() / 1e9:.2f} GB, time: {timer() - start:.2f}")
    torch.cuda.synchronize()
    end = timer()
    print("Average time per inference:", (end - start) / 5)
