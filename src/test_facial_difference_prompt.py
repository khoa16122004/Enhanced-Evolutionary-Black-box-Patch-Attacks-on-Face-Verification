import argparse
from dataset import get_dataset
from get_architech import init_lvlm_model
import torch
from PIL import Image
from tqdm import tqdm
import os

def main(args):
    dataset = get_dataset(args.dataset)
    lvlm_model, image_token, special_token = init_lvlm_model(args.pretrained, args.model_name)

    if args.return_result == 0:
        prompt = "Given the two facial images, determine whether they belong to the same person. Give the explanation for your choosing"
    else: 
        prompts = [
            "Compare the two facial images and answer with 'same' if they show the same person, or 'different' if not.",
            "Are the two provided face images from the same individual? Reply with 'same' or 'different' only.",
            "Check if both facial photos depict the same person. Return only 'same' or 'different'.",
            "Based on facial features, do the two images belong to the same person? Respond: 'same' or 'different'.",
            "Determine whether the two given face images are of the same individual. Reply using 'same' or 'different'.",
            "Do these two face images represent the same person? Output only one word: 'same' or 'different'.",
            "Assess the similarity between these two facial images. Respond only with 'same' or 'different'.",
            "Look at both facial pictures and decide: are they the same person? Answer with 'same' or 'different'.",
            "Using only facial information, determine if both images are of the same person. Return 'same' or 'different'.",
            "Are these two face photos identical in terms of identity? Reply with either 'same' or 'different'."
        ]
        # prompt = "Analyze the two provided facial images and determine if they belong to the same person. Please respond with a single digit only: '1' if you conclude they ARE the same person, and '0' if you conclude they are NOT the same person"
    with open(args.split_path, "r") as f:
        lines = [int(line.strip()) for line in f.readlines()]
    
    with torch.no_grad():
        for i, prompt in enumerate(tqdm(prompts)):
            prompt_dir = os.path.join("test_split", str(i))
            os.makedirs(prompt_dir, exist_ok=True)
            outputs = []
            for j in lines:
                img1, img2, label = dataset[j]
                print("label: ", label)
                img1.save("test_0.png")
                img2.save("test_1.png")
                input("wait")
                question = prompt + image_token * 2
                print("Question: ", question)
                response = lvlm_model.inference(question, [img1, img2])[0].replace("\n", "")
                print("Response: ", response)
                outputs.append((j, response))
                # break

            output_path = os.path.join(prompt_dir, f"{args.label}_return_result={args.return_result}_{args.pretrained}_{args.dataset}_{args.model_name}.txt")
            with open(output_path, "w") as f:
                f.write(f"{prompt}\n")
                for o in outputs:
                    f.write(f"{o[0]}\t{o[1]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_path", type=str)
    parser.add_argument("--pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--dataset", type=str, default="lfw")
    parser.add_argument("--return_result", type=int, default=0)
    parser.add_argument("--label", type=str, default="")
    args = parser.parse_args()

    main(args)
