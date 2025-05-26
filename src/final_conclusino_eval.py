import torch
import os
from dataset import get_dataset
from text_to_text_services import QwenService, GPTService, LlamaService
from tqdm import tqdm
import re


def main(args):
    dataset = get_dataset(args.dataset)
    if args.extract_llm == "qwen":
        llm = QwenService(model_name="qwen")
    elif args.extract_llm == "gpt4o":
        llm = GPTService(model_name="gpt-4o")
    else:
        llm = LlamaService(model_name="Llama-7b")

    
    acc_0 = 0
    acc_1 = 0
    system_prompt = f"""You will receive a paragraph saying whether two images show the same person or not.
    Return:
    - same if the paragraph says they are the same person ( or Yes).
    - different if they are different people (or No).
    Only return one word: same or different."""
    
    
    results = []
    
    for i in tqdm(range(len(dataset))):
        img1, img2, label = dataset[i]
        file_path = os.path.join(args.extract_dir, str(i), "decide.txt")
        with open(file_path, "r") as f:
            pred = f.read().strip()
        output = llm.text_to_text(system_prompt, pred)[0]
        print("Output: ", output)
        match = re.search(r'\b(same|different)\b', output.strip().lower())
        if match:
            results.append(match.group(1))
        else:
            print("error: ", pred)
            results.append("error")
            
        break
    with open(f"results_{args.extract_llm}_{args.extract_dir}.txt", "w") as f:
        for result in results:
            f.write(result + "\n")
        
        
        


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--extract_llm", type=str, default="Llama-7b")
    parser.add_argument("--extract_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="lfw")
    args = parser.parse_args()

    main(args)
