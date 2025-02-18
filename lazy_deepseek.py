import torch
from timeit import default_timer as timer

from transformers import PretrainedConfig
from transformers.modeling_utils import no_init_weights, init_empty_weights
from transformers.utils import ContextManagers, logging
from transformers.cache_utils import DynamicCache

from deepseek.modeling_deepseek import DeepseekV3Model
from gguf import GGUFReader
import types

from lazy_loading import (
    get_gguf_hf_weights_map,
    remove_registered_parameters,
    load_eager_module_weights,
    GLOBAL_GGUF_MAPPING,
    GLOBAL_GGUF_READER,
    lazy_load_hook,
    lazy_offload_hook,
    manual_load_hook,
    manual_offload_hook,
)


torch.manual_seed(0)
# logger = logging.get_logger(__name__)
# torch.set_grad_enabled(False)
# torch.backends.cudnn.benchmark = True
torch.set_default_dtype(torch.float16)
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.set_num_threads(12)
# torch.cuda.set_per_process_memory_fraction(0.4)
# use tf32
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# Load model configuration and create a dummy model (on "meta") for weight mapping.
pretrained_model_name_or_path = "deepseek-ai/DeepSeek-R1"
config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path)
config._attn_implementation = "flash_attention_2"
config.torch_dtype = torch.float16
with torch.device("meta"):
    dummy_model = DeepseekV3Model(config)
tensor_key_mapping = get_gguf_hf_weights_map(dummy_model)

# Load GGUF files and update the global mapping.
for i in range(1, 4):
    gguf_path = f"../DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-0000{i}-of-00003.gguf"
    # for i in range(1, 6):
    #     gguf_path = f"../DeepSeek-R1-GGUF/DeepSeek-R1-Q2_K_XS/DeepSeek-R1-Q2_K_XS-0000{i}-of-00005.gguf"
    # for i in range(1, 10):
    #     # DeepSeek-R1-Q4_K_M-00009-of-00009.gguf
    #     gguf_path = f"../DeepSeek-R1-GGUF/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-0000{i}-of-00009.gguf"
    GLOBAL_GGUF_READER = GGUFReader(gguf_path)
    # if i == 1:
    #     GGUFReader.data = np.array(GLOBAL_GGUF_READER.data)
    for tensor in GLOBAL_GGUF_READER.tensors:
        if tensor.name not in tensor_key_mapping:
            print(tensor.name, tensor.data.shape, "not in mapping")
            continue
        hf_key = tensor_key_mapping[tensor.name]
        GLOBAL_GGUF_MAPPING[hf_key] = (
            torch.from_numpy(tensor.data),
            tensor.tensor_type,
        )
        # pin memory could help with performance, though it does not work on my 3090
# Initialize the model with empty weights.
init_contexts = [no_init_weights(_enable=True), init_empty_weights()]
with ContextManagers(init_contexts):
    model = DeepseekV3Model(config)

# Remove parameters to enable lazy loading.
remove_registered_parameters(model)
for module in model.modules():
    if getattr(module, "load_once", False):
        module.register_forward_pre_hook(lazy_load_hook)
    elif hasattr(module, "lazy_params"):
        module.register_forward_pre_hook(lazy_load_hook)
        # module.register_forward_hook(lazy_offload_hook)
        module.manual_load = types.MethodType(manual_load_hook, module)
        module.manual_offload = types.MethodType(manual_offload_hook, module)

# model = torch.compile(model)
# model = torch.compile(model, fullgraph=True)
model.eval()

# Eagerly load weights for modules that should always remain on GPU.
load_eager_module_weights(model.embed_tokens, "embed_tokens")
load_eager_module_weights(model.norm, "norm")
model.embed_tokens.to("cuda")
model.norm.to("cuda")

# use lm_head tied to embed_tokens
model.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
model.lm_head.weight = model.embed_tokens.weight
# model.lm_head.to("cuda")

# warmup
with torch.no_grad():
    batch_size, seq_length = 1, 128
    input_ids = torch.randint(0, 129280, (batch_size, seq_length)).cuda()
    past_key_value = DynamicCache()
    output_attentions = False
    use_cache = True
    out = model(
        input_ids,
        past_key_values=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    x = out.last_hidden_state
    logits = model.lm_head(x)

torch.cuda.synchronize()

# --- Inference Example ---
with torch.no_grad():
    batch_size, seq_length = 1, 128
    input_ids = torch.randint(0, 129280, (batch_size, seq_length)).cuda()
    past_key_value = DynamicCache()
    output_attentions = False
    use_cache = True
    flag = True
    start = timer()
    for i in range(51):
        out = model(
            input_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        if flag:
            flag = False
            st = timer()
        x = out.last_hidden_state
        past_key_value = out.past_key_values
        # x = model.norm(x)
        logits = model.lm_head(x)
        last_token_logits = logits[:, -1, :]
        print("Final output:", logits[0, 0, :5])
        input_ids = torch.argmax(last_token_logits, dim=-1, keepdim=True)
        seq_length = 1  # new token
        print(i, timer() - start)
        start = timer()
    print("Decoding time:", (timer() - st) / 50)

start = timer()
with torch.no_grad():
    for i in range(5):
        batch_size, seq_length = 1, 128
        input_ids = torch.randint(0, 129280, (batch_size, seq_length)).cuda()
        out = model(
            input_ids,
            # past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=False,
        )
        if flag:
            flag = False
            st = timer()
        x = out.last_hidden_state
        # past_key_value = out.past_key_values
        # x = model.norm(x)
        logits = model.lm_head(x)
        print(
            f"Finished {i + 1} inference, GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB, time: {timer() - start:.2f}"
        )
        # torch.cuda.empty_cache()
torch.cuda.synchronize()
end = timer()
print("Average time per inference:", (end - start) / 5)
