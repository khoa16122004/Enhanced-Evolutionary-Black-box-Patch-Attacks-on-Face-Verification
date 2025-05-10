import argparse
from dataset import get_dataset
from get_architech import init_lvlm_model
import torch
from PIL import Image

import dotenv
dotenv.load_dotenv()

def main(args):
    dataset = get_dataset(args.dataset)
    lvlm_model, image_token, special_token = init_lvlm_model(args.pretrained, args.model_name)

    if args.return_result == 0:
        prompt = "Given the two facial images, determine whether they belong to the same person. Give the explanation for your choosing"
    else:
        prompt = "Analyze the two provided facial images and determine if they belong to the same person. Please respond with a single text only: 'Same' if you conclude they ARE the same person, and 'Difference' if you conclude they are NOT the same person"

    with torch.no_grad():
        img1 = Image.open(args.img1_path).convert("RGB")
        img2 = Image.open(args.img2_path).convert("RGB")

        question = prompt + image_token * 2
        response = lvlm_model.inference(question, [img1, img2])[0].replace("\n", "")
        print("Response: ", response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1_path", type=str)
    parser.add_argument("--img2_path", type=str)
    parser.add_argument("--pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--dataset", type=str, default="lfw")
    parser.add_argument("--return_result", type=int, default=0)
    args = parser.parse_args()

    main(args)
