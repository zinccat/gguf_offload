import torch
from timeit import default_timer as timer

from transformers import PretrainedConfig
from transformers.modeling_utils import no_init_weights, init_empty_weights
from transformers.utils import ContextManagers, logging
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.utils.import_utils import is_torch_fx_available
from transformers.cache_utils import DynamicCache

from deepseek.modeling_deepseek import DeepseekV3Model
from gguf import GGUFReader

from lazy_loading import (
    get_gguf_hf_weights_map,
    remove_registered_parameters,
    load_eager_module_weights,
    GLOBAL_GGUF_MAPPING,
    GLOBAL_GGUF_READER,
    lazy_load_hook,
    lazy_offload_hook,
)
from inference import pipelined_inference_layers


torch.manual_seed(0)
logger = logging.get_logger(__name__)
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True
# use tf32
torch.backends.cuda.matmul.allow_tf32 = True

if is_torch_fx_available():
    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)

# Load model configuration and create a dummy model (on "meta") for weight mapping.
pretrained_model_name_or_path = "deepseek-ai/DeepSeek-R1"
config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path)
# config._attn_implementation = "flash_attention_2"
with torch.device("meta"):
    dummy_model = DeepseekV3Model(config)
tensor_key_mapping = get_gguf_hf_weights_map(dummy_model)

# Load GGUF files and update the global mapping.
for i in range(1, 4):
    gguf_path = f"../DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-0000{i}-of-00003.gguf"
    GLOBAL_GGUF_READER = GGUFReader(gguf_path)
    for tensor in GLOBAL_GGUF_READER.tensors:
        if tensor.name not in tensor_key_mapping:
            print(tensor.name, tensor.data.shape, "not in mapping")
            continue
        hf_key = tensor_key_mapping[tensor.name]
        GLOBAL_GGUF_MAPPING[hf_key] = tensor
# Initialize the model with empty weights.
init_contexts = [no_init_weights(_enable=True), init_empty_weights()]
with ContextManagers(init_contexts):
    model = DeepseekV3Model(config)

# Remove parameters to enable lazy loading.
remove_registered_parameters(model)
for module in model.modules():
    if hasattr(module, "lazy_params"):
        module.register_forward_pre_hook(lazy_load_hook)
        module.register_forward_hook(lazy_offload_hook)

model.eval()

# Eagerly load weights for modules that should always remain on GPU.
load_eager_module_weights(model.embed_tokens, "embed_tokens")
load_eager_module_weights(model.norm, "norm")
model.embed_tokens.to("cuda")
model.norm.to("cuda")

for idx in range(3):
    load_eager_module_weights(model.layers[idx], f"layers.{idx}")
    model.layers[idx].to("cuda")

# use lm_head tied to embed_tokens
model.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
model.lm_head.weight = model.embed_tokens.weight
model.lm_head.to("cuda")

# --- Inference Example ---
with torch.no_grad():
    batch_size, seq_length = 1, 128
    input_ids = torch.randint(0, 129280, (batch_size, seq_length)).cuda()
    past_key_value = DynamicCache()
    output_attentions = False
    use_cache = True
    past_key_values_length = 0
    start = timer()
    for i in range(10):
        inputs_embeds = model.embed_tokens(input_ids)
        if use_cache:
            past_key_values_length = past_key_value.get_usable_length(seq_length)
        attention_mask = _prepare_4d_causal_attention_mask(
            None, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, 1, seq_length, seq_length + past_key_values_length),
                device=inputs_embeds.device,
            )
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device="cuda",
        )
        position_ids = position_ids.unsqueeze(0)
        x, past_key_value = pipelined_inference_layers(
            model.layers,
            inputs_embeds,
            chunk_size=2,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        x = model.norm(x)
        logits = model.lm_head(x)
        last_token_logits = logits[:, -1, :]
        print("Final output:", logits[0, 0, :5])
        input_ids = torch.argmax(last_token_logits, dim=-1, keepdim=True)
        seq_length = 1  # new token
        print(timer() - start)
        start = timer()

start = timer()
with torch.no_grad():
    for i in range(5):
        batch_size, seq_length = 1, 512
        input_ids = torch.randint(0, 129280, (batch_size, seq_length)).cuda()
        x = model.embed_tokens(input_ids)
        cache_position = torch.arange(0, x.shape[1], device=x.device)
        position_ids = cache_position.unsqueeze(0)
        attention_mask = _prepare_4d_causal_attention_mask(
            None, (batch_size, seq_length), x, 0
        )
        past_key_value = DynamicCache()
        output_attentions = False
        use_cache = True
        x, cache = pipelined_inference_layers(
            model.layers,
            x,
            chunk_size=6,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        x = model.norm(x)
        x = model.lm_head(x)
        print(
            f"Finished {i + 1} inference, GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB, time: {timer() - start:.2f}"
        )
torch.cuda.synchronize()
end = timer()
print("Average time per inference:", (end - start) / 5)
