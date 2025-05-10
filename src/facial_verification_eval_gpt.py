import argparse
from dataset import get_dataset
from get_architech import init_lvlm_model
import torch
from PIL import Image
import re
import os

def main(args):
    dataset = get_dataset(args.dataset)
    acc_0 = 0
    acc_1 = 0 
    num_0 = 0
    num_1 = 0   
    avg_acc = 0
    for i in range(len(dataset)):
        img1, img2, label = dataset[i]
        file_path = os.path.join(args.extracted_dir, f"{i}.txt")
        with open(file_path, "r") as f:
            pred = f.read().strip()
            pred = re.findall(r"\{\{(.*?)\}\}", pred)            
            print(pred)
        
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
    print(f"acc_0: {acc_0/num_0}, acc_1: {acc_1/num_1}, avg_acc: {avg_acc/len(num_0+num_1)}")
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted_dir", type=str)
    parser.add_argument("--dataset", type=str, default="lfw")
    args = parser.parse_args()
    main(args)

