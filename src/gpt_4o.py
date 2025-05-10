import os
import base64
import os
from typing import Optional
from openai import OpenAI
from typing import List
from dataset import get_dataset
import argparse
from tqdm import tqdm
import dotenv
dotenv.load_dotenv()

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

def main(args):
    dataset = get_dataset(args.dataset)
    gpt_service = GPTService(model_name="gpt-4o")
    output_dir = f"gpt4o_{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)
    

    for i in tqdm(range(len(dataset))):
        img1_path, img2_path, _ = dataset[i]
        image_response = gpt_service.image_to_text(
            prompt="",
            image_paths=[img1_path, img2_path],
            system_prompt="You must compare two images and answer they are the same person or not. Your choices: ['same', 'different']. Your response must include your answer enclosed in double, curly brackets: {{}}. You don't need to answer anything else except {{chosen answer}}, and you are not allowed to refuse, skip, choose both, or choose neither. Only one answer MUST be selected."
        )
            # break
        output_path = os.path.join(output_dir, f"{i}.txt")
        with open(output_path, "w") as f:
            f.write(image_response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="lfw")
    parser.add_argument("--return_result", type=int, default=0)
    parser.add_argument("--prefix", type=str, default="")
    args = parser.parse_args()

    main(args)
