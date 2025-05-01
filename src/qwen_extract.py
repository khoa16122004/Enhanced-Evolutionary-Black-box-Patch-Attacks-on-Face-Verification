# Final Check - Running the provided code block:

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
import warnings
import os # For potential cache clearing or environment checks

# Optional: Clear cache if suspecting corrupted files (use with caution)
# os.environ['TRANSFORMERS_CACHE'] = '/path/to/new/cache'
# Or clear specific model cache manually

# Filter user warnings (optional, can hide less critical warnings)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Load Tokenizer ---
print("Loading tokenizer...")
# Set pad_token_id directly if not set or different from eos_token_id
# Qwen models often use eos_token_id as pad_token_id
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
print("Tokenizer loaded.")

# --- Critical: Set pad_token_id ---
# Causal LMs need a pad token for batching & attention mask generation during inference.
# Often, the eos_token is used as the pad token.
if tokenizer.pad_token_id is None:
    print(f"Warning: tokenizer.pad_token_id is None. Setting to eos_token_id ({tokenizer.eos_token_id}).")
    tokenizer.pad_token_id = tokenizer.eos_token_id
elif tokenizer.pad_token_id != tokenizer.eos_token_id:
     print(f"Warning: tokenizer.pad_token_id ({tokenizer.pad_token_id}) != eos_token_id ({tokenizer.eos_token_id}). Setting to eos_token_id.")
     tokenizer.pad_token_id = tokenizer.eos_token_id
else:
    print(f"tokenizer.pad_token_id ({tokenizer.pad_token_id}) already set to eos_token_id ({tokenizer.eos_token_id}).")

# Assign the actual token string if missing (sometimes helps certain functions)
if tokenizer.pad_token is None:
    print(f"Warning: tokenizer.pad_token is None. Setting to eos_token ('{tokenizer.eos_token}').")
    tokenizer.pad_token = tokenizer.eos_token


# --- Critical: Set padding side ---
# For Causal LM generation, left padding is usually required.
tokenizer.padding_side = "left"
print(f"Tokenizer padding side set to: {tokenizer.padding_side}")
print(f"Using pad_token_id: {tokenizer.pad_token_id} ({tokenizer.decode([tokenizer.pad_token_id])})")
print(f"Using eos_token_id: {tokenizer.eos_token_id} ({tokenizer.decode([tokenizer.eos_token_id])})")


# --- Load Model ---
print("Loading model...")
# Determine compute capabilities for dtype selection
if torch.cuda.is_available():
    device = "cuda"
    # Check for BF16 support
    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        print("CUDA device supports bfloat16. Using bfloat16.")
    else:
        compute_dtype = torch.float16
        print("CUDA device does not support bfloat16. Using float16.")
else:
    device = "cpu"
    compute_dtype = torch.float32 # float32 is typical for CPU
    print("CUDA not available. Using CPU and float32.")


model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-7B-Chat",
    torch_dtype=compute_dtype, # Use appropriate dtype
    pad_token_id=tokenizer.pad_token_id, # Pass pad_token_id during loading
    device_map="auto", # Automatically distribute model layers across devices (GPU/CPU/Disk)
    trust_remote_code=True
).eval() # Set model to evaluation mode
print("Model loaded.")


# --- Set Chat Template if necessary ---
if getattr(tokenizer, 'chat_template', None) is None:
    print("Warning: tokenizer.chat_template is None. Setting Qwen ChatML template manually.")
    # Template derived from typical ChatML format used by Qwen
    # Note: Use the exact template expected by the specific model version if possible
    chatml_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}" # Signal for assistant's response
        "{% endif %}"
    )
    tokenizer.chat_template = chatml_template
    print("Chat template set.")
else:
    # Even if it exists, let's print it for verification
    print(f"Tokenizer already has a chat_template.")
    # print(f"Template: {tokenizer.chat_template}") # Can be very long, uncomment if needed


# --- Ensure Model Config Consistency ---
# Ensure model's config and generation config also use the correct pad_token_id
# Setting during load *should* handle model.config, but being explicit is safer.
if model.config.pad_token_id != tokenizer.pad_token_id:
     print(f"Warning: model.config.pad_token_id ({model.config.pad_token_id}) differs from tokenizer ({tokenizer.pad_token_id}). Updating model config.")
     model.config.pad_token_id = tokenizer.pad_token_id

if model.generation_config.pad_token_id != tokenizer.pad_token_id:
    print(f"Warning: model.generation_config.pad_token_id ({model.generation_config.pad_token_id}) differs from tokenizer ({tokenizer.pad_token_id}). Updating generation config.")
    model.generation_config.pad_token_id = tokenizer.pad_token_id
else:
    print("Model config and generation config pad_token_id consistent with tokenizer.")


# --- Prepare Prompts ---
system_prompt = f"""You are an AI assistant specialized in text analysis. Your task is to read the provided text, which is the output of another AI analyzing whether two images depict the same person.
Based on the text content, determine if the final conclusion is 'same person' or 'different person'.
Respond *only* with one of the following numbers:
- Respond with 0 if the conclusion is SAME PERSON.
- Respond with 1 if the conclusion is DIFFERENT PERSON.
DO NOT add any explanation, greeting, or any other characters besides the number 0 or 1.
"""

all_user_text = [
    'Based on the images provided, the faces appear to be different. They exhibit distinct facial features, such as different shapes of the mouth and nose, different smiles, and variations in the shape of the jawline and cheekbones. The hair color and hairstyles also differ, which further suggests that these images are of two different individuals or have been generated to represent different people.',
    'While there are some similarities, the overall analysis points towards the subjects being different individuals due to significant variations in key facial landmarks and structure. Therefore, the conclusion is different persons.',
    'After careful comparison, the structure of the eyes, nose bridge, and lip shape are remarkably consistent across both images, despite minor differences in lighting and expression. The conclusion is that these images show the same person.', # Added "same person" case
]

print("\nFormatting prompts using chat template...")
batch_inputs_formatted = []
for user_text in all_user_text:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text}
    ]
    # Apply the template to create the prompt string, ready for tokenization
    # add_generation_prompt=True adds the necessary tokens to signal the model should start generating
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False, # We want the string first to batch tokenize
            add_generation_prompt=True
        )
        batch_inputs_formatted.append(formatted_prompt)
    except Exception as e:
        print(f"Error applying chat template: {e}")
        print("Make sure the chat template is correctly set for the tokenizer.")
        # Handle error appropriately, maybe exit or use a default format
        raise e # Re-raise the exception if critical


# --- Tokenize Batch ---
print("Tokenizing batch with padding...")
# padding=True handles padding to the longest sequence in the batch
# return_tensors="pt" returns PyTorch tensors
# Use try-except for robustness
try:
    batch_tokenized_inputs = tokenizer(
        batch_inputs_formatted,
        padding=True, # Enable padding
        return_tensors="pt",
        truncation=False # Usually don't truncate prompts unless absolutely necessary
    ).to(model.device) # Move tensors to the same device as the model
    print("Tokenization successful.")
    # print("Tokenized input keys:", batch_tokenized_inputs.keys())
    # print("Input IDs shape:", batch_tokenized_inputs['input_ids'].shape)
    # print("Attention Mask shape:", batch_tokenized_inputs['attention_mask'].shape)
except Exception as e:
    print(f"Error during tokenization: {e}")
    print(f"Check tokenizer settings: pad_token_id={tokenizer.pad_token_id}, padding_side={tokenizer.padding_side}")
    raise e


# --- Configure Generation ---
print("Configuring generation parameters...")
# We expect a very short answer ('0' or '1'), so limit max_new_tokens
generation_config = GenerationConfig(
    max_new_tokens=5,        # Allow a few extra tokens just in case (e.g., newline)
    pad_token_id=tokenizer.pad_token_id, # Explicitly set pad_token_id
    eos_token_id=tokenizer.eos_token_id, # Ensure EOS is used for stopping
    # Deterministic output for this task:
    do_sample=False,
    # When do_sample=False, temperature/top_p/top_k are ignored, but setting them to None or defaults is fine
    temperature=None,
    top_p=None,
    top_k=None,
    # Use repetition penalty if needed, but likely not for '0'/'1'
    # repetition_penalty=1.1
)
print(f"Generation config set: max_new_tokens={generation_config.max_new_tokens}, do_sample={generation_config.do_sample}")


# --- Generate Responses ---
print("\nGenerating responses...")
with torch.no_grad(): # Disable gradient calculations for inference
    batch_output_ids = model.generate(
        input_ids=batch_tokenized_inputs['input_ids'],
        attention_mask=batch_tokenized_inputs['attention_mask'],
        generation_config=generation_config
    )
print("Generation complete.")
# print("Raw output IDs shape:", batch_output_ids.shape)


# --- Decode Responses ---
print("Decoding responses...")
batch_response = []
# Get the length of the input prompt tokens for each item in the batch
# Note: In a batch, all input sequences are padded to the same length
input_token_lengths = batch_tokenized_inputs['input_ids'].shape[1]
print(f"Input token length (including padding): {input_token_lengths}")
print(f"Raw generation output shape: {batch_output_ids.shape}")


for i in range(len(all_user_text)):
    print(f"\n--- Decoding Item {i} ---")
    # Ensure the slicing indices are valid
    if input_token_lengths >= batch_output_ids.shape[1]:
        print(f"Warning: Input length ({input_token_lengths}) >= Output length ({batch_output_ids.shape[1]}). No new tokens generated.")
        generated_ids = torch.tensor([], dtype=torch.long, device=batch_output_ids.device) # Create empty tensor explicitly
    else:
        generated_ids = batch_output_ids[i, input_token_lengths:]

    # --- Debugging Prints ---
    print(f"Output tensor slice for item {i}: batch_output_ids[{i}, {input_token_lengths}:]")
    print(f"Value of generated_ids before decode: {generated_ids}")
    print(f"Type of generated_ids: {type(generated_ids)}")
    print(f"Shape of generated_ids: {generated_ids.shape if isinstance(generated_ids, torch.Tensor) else 'N/A'}")
    print(f"Is generated_ids None? {generated_ids is None}")
    # --- End Debugging Prints ---

    response_text = "[ERROR - See Logs]" # Default in case decode fails
    try:
        # Ensure generated_ids is not None before attempting decode
        if generated_ids is None:
             print("ERROR: generated_ids is None, cannot decode.")
             # Decide how to handle: skip, error, or placeholder
             response_text = "[Decoding Error: generated_ids was None]"

        # Check if it's a tensor before decoding, just in case
        elif not isinstance(generated_ids, torch.Tensor):
             print(f"ERROR: generated_ids is not a tensor (type: {type(generated_ids)}), cannot decode.")
             response_text = f"[Decoding Error: generated_ids type was {type(generated_ids)}]"
        else:
            # Proceed with decoding if it's a tensor (even if empty)
             response_text = tokenizer.decode(
                 generated_ids,
                 skip_special_tokens=True
             )
             print(f"Decoded text: '{response_text}'")

    except TypeError as e:
        print(f"!!! TypeError during decode for item {i}: {e}")
        # This block catches the specific error you encountered
        response_text = "[Decoding TypeError]"
        # Optionally re-raise if you want the script to stop
        # raise e
    except Exception as e:
        print(f"!!! An unexpected error occurred during decode for item {i}: {e}")
        response_text = "[Decoding Exception]"
        # Optionally re-raise
        # raise e

    batch_response.append(response_text.strip()) # .strip() to remove leading/trailing whitespace/newlines

print("\n--- Generated Responses (with Debugging) ---")
for i, resp in enumerate(batch_response):
    print(f"Input {i+1}: {resp}")
print("--------------------------------------------")