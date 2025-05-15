import argparse
from dataset import get_dataset
from get_architech import init_lvlm_model
import torch
from PIL import Image
import re
from typing import Optional
import base64
import dotenv
from openai import OpenAI
import os
from typing import List
dotenv.load_dotenv()
# from gpt_4o import GTPService


class GPTService:
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize the GPTService with a model name and API key.
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in the environment variable 'OPENAI_KEY'.")
        self.client = OpenAI(api_key=self.api_key)

    def text_to_text(self, prompt: str, system_prompt: str) -> str:
        """
        Perform a text-to-text API call.
        """
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.output_text.strip()
        except Exception as e:
            print(f"Error during API call: {e}")
            return "Error occurred during API call."

    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:  
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def image_to_text(self, prompt: str, image_paths: List[str], system_prompt: str) -> str:
        """
        Perform an image-to-text API call using base64-encoded images.
        """
        try:
            base64_images = [self.encode_image(image_path) for image_path in image_paths]
            input_images = [
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}
                for b64 in base64_images
            ]

            response = self.client.responses.create(
                model=self.model_name,
                input=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}] + input_images
                    }
                ]
            )
            return response.output_text.strip()
        except Exception as e:
            print(f"Error during API call: {e}")
            return "Error occurred during API call."


def parse_response(response):
    response = response.lower()

    patterns = {
        "same": r"\b(the\s+same|same|yes|identical|matching)\b",
        "different": r"\b(different|not\s+the\s+same|no|distinct|mismatch(ed)?)\b"
    }

    for label, pattern in patterns.items():
        if re.search(pattern, response):
            return label

    # If not found, fall back to GPT
    print("Using GPT")
    client = GPTService('gpt-4o')
    system_prompt = (
        "You have received a response related to a facial verification task. "
        "Please extract the final answer from the response—such as 'same' or 'different', "
        "or equivalent expressions like 'yes' or 'no'. "
        "Normalize the extracted answer to either 'same' or 'different'."
    )
    output = client.text_to_text(system_prompt, response)
    return output.strip().lower()
    
def main(args):
    dataset = get_dataset(args.dataset)
    with open(args.extracted_path, "r") as f:
        responses = [line.strip() for line in f.readlines()]
            

    
    acc_0 = 0
    acc_1 = 0 
    num_0 = 0
    num_1 = 0   
    avg_acc = 0
    print("Len: ", len(responses))
    for i in range(len(dataset)):
        img1, img2, label = dataset[i]
        # print(i)
        pred = responses[i]
        pred = parse_response(pred)
        if pred.lower() not in ['different', 'same']:
            print("error: ", pred)
        else:
            if label == 0:
                num_0 += 1
                if pred.lower() == 'same':
                    acc_0 += 1
                    avg_acc += 1
                    
                
            else:
                num_1 += 1
                if pred.lower() == 'different':
                    acc_1 += 1
                    avg_acc += 1
                else:
                    print(i)
                    

    print("num_0: ", num_0)
    print("num_1: ", num_1)
    print(f"acc_0: {acc_0/num_0}, acc_1: {acc_1/num_1}, avg_acc: {avg_acc/len(responses)}")
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted_path", type=str)
    parser.add_argument("--dataset", type=str, default="lfw")
    args = parser.parse_args()
    main(args)

