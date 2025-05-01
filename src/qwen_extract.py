import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
import warnings

# Suppress specific warnings if needed, although it's better to address them if possible
warnings.filterwarnings("ignore", message="None of PyTorch, TensorFlow 2.0 or Flax have been found.*")
warnings.filterwarnings("ignore", message="Passing Lopad_token_id` to `generate` is deprecated.*") # Handle below


# Load tokenizer and model
# Set pad_token_id directly if not set or different from eos_token_id
# Qwen models often use eos_token_id as pad_token_id
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)

# --- Critical: Set pad_token_id ---
# Causal LMs need a pad token for batching & attention mask generation during inference.
# Often, the eos_token is used as the pad token.
if tokenizer.pad_token_id is None:
    print("Warning: pad_token_id not set. Setting to eos_token_id.")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # Ensure the model config also knows about the pad_token_id if set dynamically
    # model.config.pad_token_id = tokenizer.pad_token_id # Do this after loading model

# --- Critical: Set padding side ---
# For Causal LM generation, left padding is usually required.
tokenizer.padding_side = "left"
print(f"Tokenizer padding side set to: {tokenizer.padding_side}")
print(f"Using pad_token_id: {tokenizer.pad_token_id} ({tokenizer.decode(tokenizer.pad_token_id)})")
print(f"Using eos_token_id: {tokenizer.eos_token_id} ({tokenizer.decode(tokenizer.eos_token_id)})")


model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat",
    pad_token_id=tokenizer.pad_token_id, # Pass pad_token_id during loading
    device_map="auto",
    trust_remote_code=True
).eval()

# Ensure model's generation config also uses the correct pad_token_id
# The line below might be redundant if passing pad_token_id during loading works,
# but it's good practice to ensure consistency.
model.generation_config.pad_token_id = tokenizer.pad_token_id


# Define the system prompt
system_prompt = f"""You are an AI assistant specialized in text analysis. Your task is to read the provided text, which is the output of another AI analyzing whether two images depict the same person.
Based on the text content, determine if the final conclusion is 'same person' or 'different person'.
Respond *only* with one of the following numbers:
- Respond with 0 if the conclusion is SAME PERSON.
- Respond with 1 if the conclusion is DIFFERENT PERSON.
DO NOT add any explanation, greeting, or any other characters besides the number 0 or 1.
"""

# Input texts
all_user_text = [
    'Based on the images provided, the faces appear to be different. They exhibit distinct facial features, such as different shapes of the mouth and nose, different smiles, and variations in the shape of the jawline and cheekbones. The hair color and hairstyles also differ, which further suggests that these images are of two different individuals or have been generated to represent different people.',
    'Based on the images provided, the faces appear to be different. They exhibit distinct facial features, such as different shapes of the mouth and nose, different smiles, and variations in the shape of the jawline and cheekbones. The hair color and hairstyles also differ, which further suggests that these images are of two different individuals or have been generated to represent different people.',
    # Add a test case for "same person"
]

# Prepare batch inputs using the tokenizer's chat template
batch_inputs_formatted = []
for user_text in all_user_text:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text}
    ]
    # Apply the template to create the prompt string, ready for tokenization
    # add_generation_prompt=True adds the necessary tokens to signal the model should start generating
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False, # We want the string first to batch tokenize
        add_generation_prompt=True
    )
    batch_inputs_formatted.append(formatted_prompt)

# Tokenize the batch of formatted prompts
# padding=True handles padding to the longest sequence in the batch
# return_tensors="pt" returns PyTorch tensors
batch_tokenized_inputs = tokenizer(
    batch_inputs_formatted,
    padding=True,
    return_tensors="pt"
).to(model.device)


# Configure generation parameters
# We expect a very short answer ('0' or '1'), so limit max_new_tokens
generation_config = GenerationConfig(
    max_new_tokens=5, # Allow a few extra tokens just in case (e.g., newline)
    pad_token_id=tokenizer.pad_token_id, # Explicitly set pad_token_id
    eos_token_id=tokenizer.eos_token_id, # Ensure EOS is used
    # You might want deterministic output for this task:
    do_sample=False,
    temperature=None, # Often better to leave as None if do_sample=False
    top_p=None,
    # Use model's default config for other params unless needed
    # **model.generation_config.to_dict() # Use model defaults and override specific ones
)


# Generate responses
with torch.no_grad():
    batch_output_ids = model.generate(
        input_ids=batch_tokenized_inputs['input_ids'],
        attention_mask=batch_tokenized_inputs['attention_mask'],
        generation_config=generation_config
    )

# Decode the generated tokens, skipping the prompt part
batch_response = []
# Get the length of the input prompt tokens for each item in the batch
input_token_lengths = batch_tokenized_inputs['input_ids'].shape[1]

for i in range(len(all_user_text)):
    # Slice the output_ids to get only the generated part
    generated_ids = batch_output_ids[i, input_token_lengths:]
    # Decode the generated tokens
    response_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )
    batch_response.append(response_text.strip()) # .strip() to remove leading/trailing whitespace

print(batch_response)