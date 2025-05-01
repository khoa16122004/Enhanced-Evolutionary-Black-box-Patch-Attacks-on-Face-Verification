import os
import re
import google.generativeai as genai
import dotenv
from typing import Optional # Optional is needed for type hinting 'int | None' in older Python versions
                           # Use 'int | None' directly if using Python 3.10+
import argparse

# --- Constants (Optional, but can make configuration clearer) ---
GEMINI_MODEL_NAME = 'gemini-1.5-flash'
ENV_VAR_API_KEY = "GOOGLE_API_KEY"

def analyze_similarity_text_with_gemini(llm_output_text: str) -> int | None:
    """
    Analyzes text describing image similarity using Google Gemini to determine
    if the conclusion is 'same person' (0) or 'different person' (1).

    This function handles:
    1. Loading environment variables (.env file).
    2. Retrieving the Google API Key.
    3. Initializing the Google Gemini client and model.
    4. Constructing a prompt for Gemini.
    5. Calling the Gemini API.
    6. Parsing the response to extract 0 or 1.

    Args:
        llm_output_text: The text output from a previous LLM (or any text
                         describing image similarity).

    Returns:
        0 if the text concludes it's the same person.
        1 if the text concludes it's a different person.
        None if unable to determine, API key is missing/invalid,
             input text is empty, or another error occurs.

    Note:
        This function initializes the Gemini client on each call. For frequent use
        within an application, consider initializing the client once outside
        this function and passing the 'model' object as an argument for efficiency.
    """
    # --- Environment Setup ---
    # Load environment variables from a .env file if it exists
    dotenv.load_dotenv()

    # --- API Key Handling ---
    api_key = os.getenv(ENV_VAR_API_KEY)
    if not api_key:
        print(f"Error: {ENV_VAR_API_KEY} environment variable not set.")
        print(f"Please create a .env file with {ENV_VAR_API_KEY}='your-api-key-here'")
        return None # Cannot proceed without API key

    # --- Initialize Google Gemini Client ---
    # This happens on every call - see Note in docstring about efficiency
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception as e:
        print(f"Error configuring Google Generative AI or invalid API key: {e}")
        return None # Return None on initialization error

    # --- Input Validation ---
    if not llm_output_text:
        print("Error: Input text (llm_output_text) is empty.")
        return None

    # --- Prepare Prompt for Gemini ---
    prompt = f"""You are an AI assistant specialized in text analysis. Your task is to read the provided text, which is the output of another AI analyzing whether two images depict the same person.
Based on the text content, determine if the final conclusion is 'same person' or 'different person'.
Respond *only* with one of the following numbers:
- Respond with 0 if the conclusion is SAME PERSON.
- Respond with 1 if the conclusion is DIFFERENT PERSON.
DO NOT add any explanation, greeting, or any other characters besides the number 0 or 1.

--- Text to Analyze ---
{llm_output_text}
--- End of Text ---

Result (output 0 or 1 only):"""

    # --- Call Gemini API and Process Response ---
    try:
        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            temperature=0.0,      # Set temperature to 0 for deterministic output
            max_output_tokens=5   # Limit output tokens to just the number
        )

        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )

        # Check if the response was blocked or empty
        if not response.parts:
            block_reason = "Unknown"
            safety_ratings = "Not available"
            if hasattr(response, 'prompt_feedback'):
                 if hasattr(response.prompt_feedback, 'block_reason'):
                    block_reason = response.prompt_feedback.block_reason
                 if hasattr(response.prompt_feedback, 'safety_ratings'):
                    safety_ratings = response.prompt_feedback.safety_ratings

            print(f"Error: Gemini API did not return text.")
            print(f"  Block Reason: {block_reason}")
            print(f"  Safety Ratings: {safety_ratings}")
            return None

        # Extract and clean the response text
        extracted_text = response.text.strip()

        # --- Parse the Extracted Text ---
        if extracted_text == "0":
            return 0
        elif extracted_text == "1":
            return 1
        else:
            # Fallback: Try to extract 0 or 1 using regex if the output isn't exact
            numbers_found = re.findall(r'\b([01])\b', extracted_text)
            if len(numbers_found) == 1:
                 print(f"Warning: Gemini API returned '{extracted_text}', but extracted the number '{numbers_found[0]}' using regex.")
                 return int(numbers_found[0])

            # If still not found or ambiguous
            print(f"Error: Gemini API returned an unexpected result: '{extracted_text}'. Cannot determine 0 or 1.")
            return None

    except Exception as e:
        print(f"An error occurred while calling the Google Gemini API or processing the result: {e}")
        return None


def main(args):
    with open(args.response_path, "r") as f:
        responses = [line.strip() for line in f.readlines()]
    
    output_path = args.response_path.replace(".txt", "_processed.txt")
    with open(output_path, "w") as f:
        for response in responses:
            processed_response = analyze_similarity_text_with_gemini(response)
            f.write(f"{processed_response}\n")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_path", type=str, default="")
    args = parser.parse_args()
    main(args)