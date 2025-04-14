import argparse
from dataset import get_dataset
from get_architech import init_lvlm_model, get_face_encoder
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F


toTensor = transforms.ToTensor()

def main(args):
    dataset = get_dataset(args.dataset)
    lvlm_model, image_token, special_token = init_lvlm_model(args.pretrained, args.model_name)
    prompt ="Given the two facial images, let me know if they are the same person or not, in the following format: 0 for the same person, 1 for not the same person. Facial images:"
    # prompt ="Given the two facial images, let me know if they are the same person or not, explain your answer for details. Facial images:"
    acc_0 = 0
    acc_1 = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            img1, img2, label = dataset[i]
            img1, img2 = img1.resize((224, 224)), img2.resize((224, 224))
            
            qs = prompt + image_token * 2
            output = lvlm_model.inference(qs, [img1, img2])
            if output[0] == str(label):
                if label == 0:
                    acc_0 += 1
                else:
                    acc_1 += 1
                

    output_path = f"{args.pretrained}_{args.dataset}_{args.model_name}.txt"
    with open(output_path, "w") as f:
        f.write(f"acc_0: {acc_0 / len(dataset)}\n")
        f.write(f"acc_1: {acc_1 / len(dataset)}\n")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--log_output_path", type=str)
    parser.add_argument("--dataset", type=str, default="lfw")
    args = parser.parse_args()
    
    main(args)