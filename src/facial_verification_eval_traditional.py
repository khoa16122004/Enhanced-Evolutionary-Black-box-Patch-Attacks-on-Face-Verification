import argparse
from dataset import get_dataset
from get_architech import get_face_encoder
import torch
from PIL import Image
from torchvision import transforms  
import torch.nn.functional as F

def main(args):
    model, img_size = get_face_encoder(args.model_name)

    transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor(),
                                    ])
    dataset = get_dataset(args.dataset, transform)

    outputs = []

    with torch.no_grad():
        for i in range(len(dataset)):
            img1, img2, _ = dataset[i]
            img1_embedding = model(img1.unsqueeze(0).cuda())
            img2_embedding = model(img2.unsqueeze(0).cuda())
            
            img1_embedding = F.normalize(img1_embedding, p=2, dim=1)
            img2_embedding = F.normalize(img2_embedding, p=2, dim=1)
            
            sim = img1_embedding @ img2_embedding.T
            print("sim: ", sim)
            if sim >= args.threshold:
                response = 0
            else:
                response = 1

            outputs.append(response)
            # print("Response: ", response)
            # break

    output_path = f"original_image_{args.model_name}_{args.dataset}.txt"
    with open(output_path, "w") as f:
        for o in outputs:
            f.write(f"{o}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--dataset", type=str, default="lfw")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    main(args)
