import torch
from timeit import default_timer as timer

from transformers import PretrainedConfig, AutoTokenizer, TextStreamer
from transformers.modeling_utils import no_init_weights, init_empty_weights
from transformers.utils import ContextManagers
from transformers.cache_utils import DynamicCache

from deepseek.modeling_deepseek import DeepseekV3ForCausalLM
from gguf import GGUFReader
import types

from lazy_loading import (
    get_gguf_hf_weights_map,
    remove_registered_parameters,
    load_eager_module_weights,
    GLOBAL_GGUF_MAPPING,
    GLOBAL_GGUF_READER,
    lazy_load_hook,
    manual_load_hook,
    manual_offload_hook,
)

torch.manual_seed(0)
torch.set_default_dtype(torch.bfloat16)
# torch.cuda.set_per_process_memory_fraction(0.4)

# Load model configuration and create a dummy model (on "meta") for weight mapping.
pretrained_model_name_or_path = "deepseek-ai/DeepSeek-R1"
config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path)
config._attn_implementation = "flash_attention_2"
config.torch_dtype = torch.bfloat16
with torch.device("meta"):
    dummy_model = DeepseekV3ForCausalLM(config)
tensor_key_mapping = get_gguf_hf_weights_map(dummy_model.model)

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
            if tensor.name == "output.weight":
                GLOBAL_GGUF_MAPPING["lm_head.weight"] = (
                    torch.from_numpy(tensor.data),
                    tensor.tensor_type,
                )
            else:
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
    model = DeepseekV3ForCausalLM(config)

# Remove parameters to enable lazy loading.
remove_registered_parameters(model.model)
for module in model.model.modules():
    if getattr(module, "load_once", False):
        module.register_forward_pre_hook(lazy_load_hook)
    elif hasattr(module, "lazy_params"):
        module.register_forward_pre_hook(lazy_load_hook)
        # module.register_forward_hook(lazy_offload_hook)
        module.manual_load = types.MethodType(manual_load_hook, module)
        module.manual_offload = types.MethodType(manual_offload_hook, module)

model.eval()

# Eagerly load weights for modules that should always remain on GPU.
load_eager_module_weights(model.model.embed_tokens, "embed_tokens")
load_eager_module_weights(model.model.norm, "norm")
load_eager_module_weights(model.lm_head, "lm_head")
model.model.embed_tokens.to("cuda")
model.model.norm.to("cuda")
model.lm_head.to("cuda")

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

st = timer()
prompt = "<｜User｜>1+1等于几<｜Assistant｜>"
streamer = TextStreamer(tokenizer)
past_key_value = DynamicCache()
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
generate_ids = model.generate(
    inputs.input_ids,
    streamer=streamer,
    max_length=30,
    past_key_values=past_key_value,
)
print(generate_ids)
print(len(inputs.input_ids[0]), len(generate_ids[0]))
print(
    tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
)
print("Time:", timer() - st)
