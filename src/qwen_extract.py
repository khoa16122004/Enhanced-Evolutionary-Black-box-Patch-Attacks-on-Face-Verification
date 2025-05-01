from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'device: {device}')

# Load the tokenizer with left-padding
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", padding_side='left')

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-7B-Chat",
    torch_dtype="auto",
    device_map="auto"
)

# List of prompts for batch inference
prompts = [
    "Give me a short introduction to large language models.",
    "What is the capital of France?",
    "Explain the process of photosynthesis.",
    "What are the benefits of machine learning?"
]

# Messages template for each prompt
messages_template = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": None}  # Placeholder for prompt
]

# Apply chat template to each prompt and tokenize
texts = [
    tokenizer.apply_chat_template(
        [messages_template[0], {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    ) for prompt in prompts
]

# Tokenize the batch of formatted prompts
model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

# Generate responses in batch
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512, # 512 recommended
)

# Extract generated tokens corresponding to each input prompt
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# Decode the generated responses
responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

# Print each response
for i, response in enumerate(responses):
    print(f"Response to prompt {i + 1}: {response}")
    print('-'*50)
