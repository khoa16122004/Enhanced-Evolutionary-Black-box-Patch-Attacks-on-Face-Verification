import argparse
from dataset import get_dataset
from get_architech import init_lvlm_model, get_face_encoder
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

def ii_fo(img1, img2, facial_encode_model, steps=80, alpha=0.01, epsilon=0.01):
    # img1, img2: [0, 1]
    
    img1_embedding = facial_encode_model(img1)
    img2_embedding = facial_encode_model(img2)
    img1_ = img1.clone().detach()
    
    print("orignal sim: ", torch.sum(img1_embedding * img2_embedding, dim=1))

    delta = torch.zeros_like(img1_, requires_grad=True)
    
    for step in range(steps):
        image_adv = img1_ + delta
        clean_image_embedding = facial_encode_model(image_adv)
        loss = torch.sum(clean_image_embedding * img2_embedding, dim=1)
        loss.backward()
        gradient = delta.grad.detach()
        delta_data = torch.clamp(delta - alpha * torch.sign(gradient), -epsilon, epsilon)
        delta.data = delta_data
        delta.grad.zero_()
        print("Loss: ")
    
    img1_adv = img1_ + delta
    adv_embedding = facial_encode_model(image_adv)
    print("adv sim: ", torch.sum(adv_embedding * img2_embedding, dim=1))
    
        
    return img1_adv

toTensor = transforms.ToTensor()
def main(args):
    dataset = get_dataset(args.dataset)
    face_encoder = get_face_encoder("restnet_vggface")
    
    img1, img2, label = dataset[args.index]
    img1, img2 = img1.resize((160, 160)), img2.resize((160, 160))
    img1_tensor, img2_tensor = toTensor(img1).cuda(), toTensor(img2).cuda()
    img1_adv = ii_fo(img1_tensor, img2_tensor, face_encoder) # torch tensor
    img_adv_pil = transforms.ToPILImage()(img1_adv)
    img_adv_pil.save("test_fo.png")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="lfw")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    
    main(args)
    
    