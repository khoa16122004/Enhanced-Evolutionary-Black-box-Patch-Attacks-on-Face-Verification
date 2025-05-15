import argparse
from dataset import get_dataset
from get_architech import init_lvlm_model
import torch
from PIL import Image
import re
from gpt_4o import GTPService

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
    client = GTPService('gpt-4o')
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

