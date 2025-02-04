import torch
from lazy_loading import lazy_load_hook, lazy_offload_hook

def chunked_layers(layers, chunk_size=4):
    layers_list = list(layers)
    return [layers_list[i : i + chunk_size] for i in range(0, len(layers_list), chunk_size)]

def clone_module(module):
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
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
    with torch.no_grad():
        for layer in chunk:
            x = layer(x, **kwargs)[0]
    return x

def pipelined_inference_layers(layers, x, chunk_size=4, **kwargs):
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

    compute_done_events = [torch.cuda.Event(enable_timing=False) for _ in range(num_chunks)]
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
            y = run_chunk(gpu_chunks[i], y, **kwargs)
            compute_done_events[i].record(comp_stream)
        if i + 1 < num_chunks:
            with torch.cuda.stream(load_stream):
                gpu_chunks[i + 1] = copy_layers_to_device(net_chunks[i + 1], device, non_blocking=True)
                load_done_events[i + 1] = torch.cuda.Event(enable_timing=False)
                load_done_events[i + 1].record(load_stream)
        with torch.cuda.stream(cleanup_stream):
            cleanup_stream.wait_event(compute_done_events[i])
            gpu_chunks[i] = None  # Free GPU memory.
    comp_stream.synchronize()
    load_stream.synchronize()
    cleanup_stream.synchronize()

    return y
