import argparse
from dataset import get_dataset
from get_architech import init_lvlm_model
import torch
from PIL import Image

def main(args):
    dataset = get_dataset(args.dataset)
    lvlm_model, image_token, special_token = init_lvlm_model(args.pretrained, args.model_name)

    if args.return_result == 0:
        prompt = "Given the two facial images, determine whether they belong to the same person. Give the explanation for your choosing"
    else: 
        prompt = "Analyze the two provided facial images and determine if they belong to the same person. Please respond with a single text only: 'same' if you conclude they ARE the same person, and 'difference' if you conclude they are NOT the same person"
    outputs = []

    with torch.no_grad():
        for i in range(len(dataset)):
            img1, img2, _ = dataset[i]
            # img1 = img1.resize((224, 224))
            # img2 = img2.resize((224, 224))

            question = prompt + image_token * 2
            print("Question: ", question)
            response = lvlm_model.inference(question, [img1, img2])[0].replace("\n", "")
            outputs.append(response)
            print("Response: ", response)
            # break

    output_path = f"{args.prefix}_return_result={args.return_result}_{args.pretrained}_{args.dataset}_{args.model_name}.txt"
    with open(output_path, "w") as f:
        for o in outputs:
            f.write(f"{o}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--dataset", type=str, default="lfw")
    parser.add_argument("--return_result", type=int, default=0)
    parser.add_argument("--prefix", type=str, default="")
    args = parser.parse_args()

    main(args)
