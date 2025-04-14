import argparse
from dataset import get_dataset
from get_architech import init_lvlm_model
import torch
from PIL import Image
def main(args):
    dataset = get_dataset(args.dataset)
    lvlm_model, image_token, special_token = init_lvlm_model(args.pretrained, args.model_name)
    
    prompt ="Given the two facial images, let me know if they are the same person or not, in the following format: 0 for the same person, 1 for not the same person. Facial images:"
    # prompt ="Given the two facial images, let me know if they are the same person or not, explain your answer for details. Facial images:"
    acc = 0
    
    with torch.no_grad():
        for i in range(len(dataset)):
            img1, img2, label = dataset[i]
            if args.test == 1:
                img_test = input("Your path: ")
                img1 = Image.open(img_test).convert("RGB")
            
            # img1, img2 = img1.resize((224, 224)), img2.resize((224, 224))
            img1.save("img1.jpg")
            img2.save("img2.jpg")
            qs = prompt + image_token * 2
            print("question: ", qs)
            output = lvlm_model.inference(qs, [img1, img2])
            print("Output: ", output[0])
            print("Label: ", label)
            break
            if output[0] == str(label):
                acc += 1
                
    print("Accuracy: ", acc / len(dataset))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--log_output_path", type=str)
    parser.add_argument("--dataset", type=str, default="lfw")
    parser.add_argument("--test", type=int, default=0)
    args = parser.parse_args()
    
    main(args)