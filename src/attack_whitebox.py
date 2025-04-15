import argparse
from dataset import get_dataset
from get_architech import init_lvlm_model, get_face_encoder
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def ii_fo(img1, img2, facial_encode_model, steps=80, alpha=0.01, epsilon=0.01):
    facial_encode_model.eval()

    img1 = img1.detach().clone().unsqueeze(0).to(torch.float32)  # [1, C, H, W]
    img2 = img2.detach().clone().unsqueeze(0).to(torch.float32)

    img1_adv = img1.clone().detach().requires_grad_(True)

    with torch.no_grad():
        target_feat = facial_encode_model(img2)
        target_feat = F.normalize(target_feat, dim=1)

    for _ in range(steps):
        output_feat = facial_encode_model(img1_adv)
        output_feat = F.normalize(output_feat, dim=1)

        loss = F.cosine_similarity(output_feat, target_feat, dim=1).mean() # đang cần giảm

        img1_adv.grad = None
        loss.backward()

        with torch.no_grad():
            img1_adv += alpha * img1_adv.grad.sign()
            perturbation = torch.clamp(img1_adv - img1, min=-epsilon, max=epsilon)
            img1_adv = torch.clamp(img1 + perturbation, 0, 1).detach_().requires_grad_(True)

    return img1_adv.squeeze(0)


toTensor = transforms.ToTensor()
def main(args):
    dataset = get_dataset(args.dataset)
    face_encoder = get_face_encoder("restnet_vggface")
    
    img1, img2, label = dataset[args.index]
    img1, img2 = img1.resize((160, 160)), img2.resize((160, 160))
    img1_tensor, img2_tensor = toTensor(img1).unsqueeze(0).cuda(), toTensor(img2).unsqueeze(0).cuda()
    img1_adv = ii_fo(img1_tensor, img2_tensor, face_encoder) # torch tensor
    img_adv_pil = transforms.ToPILImage()(img1_adv)
    img_adv_pil.save("test_fo.png")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="lfw")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    
    main(args)
    
    