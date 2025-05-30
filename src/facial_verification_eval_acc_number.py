import argparse
from dataset import get_dataset
from get_architech import init_lvlm_model
import torch
from PIL import Image
import re





def main(args):
    dataset = get_dataset(args.dataset)
    with open(args.extracted_path, "r") as f:
        responses = []
        for i, line in enumerate(f):
            
            match = re.search(r"\b\d+\b", line.strip())
            if match:
                responses.append(int(match.group()))
            else:
                print(f"error: {i}", line.strip())
            # responses.append(line.strip())
    
    acc_0 = 0
    acc_1 = 0 
    num_0 = 0
    num_1 = 0   
    avg_acc = 0
    print("Len: ", len(responses))
    for i in range(len(dataset)):
        img1, img2, label = dataset[i]
        # print(i)
        pred = int(responses[i])
        
        if label == 0:
            num_0 += 1
            if pred == 0:
                acc_0 += 1
                avg_acc += 1
        else:
            num_1 += 1
            if pred == 1:
                acc_1 += 1
                avg_acc += 1
                    

    print("num_0: ", num_0)
    print("num_1: ", num_1)
    print(f"acc_0: {acc_0/num_0}, acc_1: {acc_1/num_1}, avg_acc: {avg_acc/len(responses)}")
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted_path", type=str)
    parser.add_argument("--dataset", type=str, default="lfw")
    args = parser.parse_args()
    main(args)

