import torch
from lazy_loading import lazy_load_hook, lazy_offload_hook

from typing import Optional, Tuple


def chunked_layers(layers, chunk_size=4):
    layers_list = list(layers)
    return [
        layers_list[i : i + chunk_size] for i in range(0, len(layers_list), chunk_size)
    ]


def clone_module(module):
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    # remove hooks
    # clone._forward_pre_hooks = {}
    # clone._forward_hooks = {}
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


# def copy_layers_to_device(layers, device, non_blocking=True):
#     new_layers = []

#     # Create a CUDA stream for asynchronous operations
#     prefetch_stream = torch.cuda.Stream()

#     for layer in layers:
#         new_layer = clone_module(layer)

#         with torch.cuda.stream(prefetch_stream):
#             # for attr, hf_key in getattr(new_layer, "lazy_params", {}).items():
#             for module in new_layer.modules():
#                 # hf_key = hf_key.rsplit(".", 1)[0]
#                 # expert_idx = None
#                 # module = get_module_by_name(new_layer, hf_key)
#                 if getattr(module, "lazy_params", None) is None:
#                     continue
#                 for attr, hf_key in module.lazy_params.items():
#                     param = getattr(module, attr)
#                     # Check if the parameter is not loaded yet and needs to be fetched
#                     if param is None or (hasattr(param, "device") and param.device.type == "meta"):
#                         map_key = hf_key
#                         if "experts" in map_key and "shared_experts" not in map_key:
#                             continue
#                             match = re.search(r"mlp.experts.(\d+)", hf_key)
#                             if match:
#                                 expert_idx = int(match.group(1))
#                                 hf_key = re.sub(r"mlp.experts.(\d+)", "mlp.experts", hf_key)
#                             else:
#                                 expert_idx = None
#                         else:
#                             expert_idx = None

#                         if map_key.endswith("e_score_correction_bias"):
#                             map_key = map_key.replace("e_score_correction_bias", "e_score_correction.bias")

#                         # Fetch weights from GLOBAL_GGUF_MAPPING
#                         if map_key not in GLOBAL_GGUF_MAPPING:
#                             raise ValueError(f"GGUF mapping does not contain key: {map_key}")
#                         else:
#                             gguf_tensor = GLOBAL_GGUF_MAPPING[map_key]
#                         # print(map_key, id(gguf_tensor.data))
#                         # Dequantize and move tensor to device asynchronously
#                         param_tensor = dequantize(gguf_tensor.data, gguf_tensor.tensor_type, expert_idx)
#                         # setattr(new_layer, attr, param_tensor.to(device, non_blocking=non_blocking))
#                         hf_key = hf_key.split(".", 2)[-1]
#                         # setattr(new_layer, hf_key.split(".", 2)[-1], param_tensor.to(device, non_blocking=non_blocking))
#                         # setattr by traversing the module hierarchy
#                         parts = hf_key.split(".")
#                         current_module = new_layer
#                         last_part = ""
#                         for part in parts[:-1]:
#                             if last_part == "experts":
#                                 current_module = current_module[expert_idx]
#                             else:
#                                 current_module = getattr(current_module, part)
#                             last_part = part
#                         setattr(current_module, parts[-1], param_tensor.to(device, non_blocking=non_blocking))

#         # Now that weights are prefetched, move the layer to the GPU asynchronously
#         new_layers.append(new_layer.to(device, non_blocking=non_blocking))

#     # Synchronize the prefetch stream to ensure all operations complete before returning
#     prefetch_stream.synchronize()

#     return new_layers


def run_chunk(
    chunk,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    with torch.no_grad():
        for layer in chunk:
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
    return layer_outputs


def pipelined_inference_layers(
    layers,
    hidden_states: torch.Tensor,
    chunk_size=4,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    x = hidden_states
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

    with torch.cuda.stream(comp_stream):
        y = x.to(device, non_blocking=True)

    with torch.cuda.stream(load_stream):
        gpu_chunks[0] = copy_layers_to_device(net_chunks[0], device, non_blocking=True)
        load_done_events[0] = torch.cuda.Event(enable_timing=False)
        load_done_events[0].record(load_stream)

    for i in range(num_chunks):
        comp_stream.wait_event(load_done_events[i])
        with torch.cuda.stream(comp_stream):
            layer_output = run_chunk(
                gpu_chunks[i],
                hidden_states=y,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            y = layer_output[0]
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
            gpu_chunks[i] = None  # Free GPU memory.
    # comp_stream.synchronize()
    # load_stream.synchronize()
    # cleanup_stream.synchronize()

    if use_cache:
        return y, layer_output[1]

    # torch.cuda.empty_cache()
    return y
