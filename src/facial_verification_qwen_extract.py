import argparse
from qwen.model import QWENModel
from tqdm import tqdm
from gpt_4o import GPTService
import os

import dotenv
dotenv.load_dotenv()

def main(args):
    
    gpt_service = GPTService(model_name="gpt-4o")
    system_prompt = f"""You will receive a paragraph saying whether two images show the same person or not.
    Return:
    - 0 if they are the same person (Yes).
    - 1 if they are different people (No).
    Only return one character: 0 or 1."""


    
    with open(args.response_path, "r") as f:
        responses = [line.strip() for line in f.readlines()]
    
    output_path = args.response_path.replace(".txt", "_processed.txt")
    
    
    
    with open(output_path, "w") as f:
        for line in tqdm(responses):
            output = gpt_service.text_to_text(prompt=f"Paragraph: {line}", 
                                               system_prompt=system_prompt).strip()
            print("Response: ", output)
            # input()
            f.write(output + "\n")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default="8")
    args = parser.parse_args()
    main(args)