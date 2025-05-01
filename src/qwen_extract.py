import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from qwen_generation_utils import make_context, decode_tokens, get_stop_words_ids

# To generate attention masks automatically, it is necessary to assign distinct
# token_ids to pad_token and eos_token, and set pad_token_id in the generation_config.
tokenizer = AutoTokenizer.from_pretrained(
    'Qwen/Qwen-7B-Chat',
    pad_token='<|extra_0|>',
    eos_token='<|endoftext|>',
    padding_side='left',
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen-7B-Chat',
    pad_token_id=tokenizer.pad_token_id,
    device_map="auto",
    trust_remote_code=True
).eval()
model.generation_config = GenerationConfig.from_pretrained('Qwen/Qwen-7B-Chat', pad_token_id=tokenizer.pad_token_id)

all_raw_text = ["Based on the images provided, the faces appear to be different. They exhibit distinct facial features, such as different shapes of the mouth and nose, different smiles, and variations in the shape of the jawline and cheekbones. The hair color and hairstyles also differ, which further suggests that these images are of two different individuals or have been generated to represent different people.", 
                "Based on the images provided, the faces appear to be different. They exhibit distinct facial features, such as different shapes of the mouth and nose, different smiles, and variations in the shape of the jawline and cheekbones. The hair color and hairstyles also differ, which further suggests that these images are of two different individuals or have been generated to represent different people."]

batch_raw_text = []
system_prompt = f"""You are an AI assistant specialized in text analysis. Your task is to read the provided text, which is the output of another AI analyzing whether two images depict the same person.
Based on the text content, determine if the final conclusion is 'same person' or 'different person'.
Respond *only* with one of the following numbers:
- Respond with 0 if the conclusion is SAME PERSON.
- Respond with 1 if the conclusion is DIFFERENT PERSON.
DO NOT add any explanation, greeting, or any other characters besides the number 0 or 1.
"""


for q in all_raw_text:
    raw_text, _ = make_context(
        tokenizer,
        q,  
        system=system_prompt,
        max_window_size=model.generation_config.max_window_size,
        chat_format=model.generation_config.chat_format,
    )
    batch_raw_text.append(raw_text)

batch_input_ids = tokenizer(batch_raw_text, padding='longest')
batch_input_ids = torch.LongTensor(batch_input_ids['input_ids']).to(model.device)
batch_out_ids = model.generate(
    batch_input_ids,
    return_dict_in_generate=False,
    generation_config=model.generation_config
)
padding_lens = [batch_input_ids[i].eq(tokenizer.pad_token_id).sum().item() for i in range(batch_input_ids.size(0))]

batch_response = [
    decode_tokens(
        batch_out_ids[i][padding_lens[i]:],
        tokenizer,
        raw_text_len=len(batch_raw_text[i]),
        context_length=(batch_input_ids[i].size(0)-padding_lens[i]),
        chat_format="chatml",
        verbose=False,
        errors='replace'
    ) for i in range(len(all_raw_text))
]
print(batch_response)

response, _ = model.chat(tokenizer, "我想听你说爱我。", history=None)
print(response)

response, _ = model.chat(tokenizer, "今天我想吃点啥，甜甜的，推荐下", history=None)
print(response)

response, _ = model.chat(tokenizer, "我马上迟到了，怎么做才能不迟到", history=None)
print(response)