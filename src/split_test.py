from PIL import Image
import os
from dataset import get_dataset
import torch


dataset = get_dataset("lfw")
num_each = 20
nums_0 = []
nums_1 = []
output_path = f"split_{num_each}_lfw_"
for i in range(len(dataset)):
    img1, img2, label = dataset[i]
    if label == 0:
        if len(nums_0) == num_each:
            continue
        nums_0.append(i)
    else:
        if len(nums_1) == num_each:
            continue
        nums_1.append(i)
        
with open(output_path + "same.txt", "w") as f:
    for i in range(num_each):
        f.write(f"{nums_0[i]}\n")
with open(output_path + "different.txt", "w") as f:
    for i in range(num_each):
        f.write(f"{nums_1[i]}\n")
        
        

        

