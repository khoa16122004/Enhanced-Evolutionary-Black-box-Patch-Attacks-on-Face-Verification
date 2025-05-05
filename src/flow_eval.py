from get_architech import init_lvlm_model
import torch
from PIL import Image
import os
from dataset import get_dataset

class Agent:
    def __init__(self, lvlm, lvlm_image_token,
                 eval_lvlm, eval_lvlm_image_token,
                 steps):
        self.lvlm = lvlm
        self.steps = steps
        self.eval_lvlm = eval_lvlm
        self.lvlm_image_token = lvlm_image_token
        self.eval_lvlm_image_token = eval_lvlm_image_token

    def eval(self, img_files, temperature=0.8):

        prompt_base = (
            "You are shown two facial images. Your task is to carefully identify any significant differences across the following biometric traits:\n"
            "- Eyes: shape, size, eyelids, wrinkles, spacing\n"
            "- Nose: length, width, nostrils, bridge\n"
            "- Mouth: lip thickness, symmetry, corners\n"
            "- Face shape: jaw width, chin structure, facial symmetry\n"
            "- Eyebrows: shape, density, gap between brows\n\n"
            "Be skeptical and assume they are **different people** unless overwhelming evidence proves otherwise. Avoid vague statements. Only conclude 'same person' if features match precisely.\n\n"
            "Images:"
        )

        full_prompt = prompt_base + self.lvlm_image_token * 2
        outputs = self.lvlm.inference(
            full_prompt, img_files,
            num_return_sequences=10, do_sample=True,
            temperature=temperature, reload=False
        )

        print(f"Generated {len(outputs)} descriptions:")
        combined_descriptions = ""
        for i, output in enumerate(outputs):
            print(f"Description {i+1}:\n{output}\n")
            combined_descriptions += f"Description {i+1}:\n{output}\n\n"

        summary_prompt = (
            "You are given multiple reasoning descriptions based on two facial images. "
            "Some descriptions may conflict. Pay close attention to the details and disagreement. "
            "Then conclude: are these **exactly the same person**, **likely the same**, or **clearly different**?\n\n"
            f"{combined_descriptions}"
            "Images:"
        )

        final_response = self.lvlm.inference(
            summary_prompt + self.lvlm_image_token * 2,
            img_files, num_return_sequences=1,
            do_sample=False, temperature=0, reload=False
        )

        # print("Final decision:\n", final_response[0])
        return final_response[0]


def main(args):
    dataset = get_dataset(args.dataset)
    lvlm_model, lvlm_image_token, lvlm_special_token = init_lvlm_model(args.lvlm_pretrained, args.lvlm_model_name)
    eval_lvlm, eval_lvlm_image_token, eval_lvlm_special_token = init_lvlm_model(args.eval_lvlm_pretrained, args.eval_lvlm_model_name)
    
    with torch.no_grad():
        img1 = Image.open(args.img1_path).convert("RGB")
        img2 = Image.open(args.img2_path).convert("RGB")

        agent = Agent(lvlm_model, lvlm_image_token,
                      eval_lvlm, eval_lvlm_image_token,
                      args.steps)


        with open(args.split_path, "r") as f:
            lines = [int(line.strip()) for line in f.readlines()]
    
        with torch.no_grad():
            prompt_dir = os.path.join("test_split", 'agent')
            os.makedirs(prompt_dir, exist_ok=True)
            outputs = []
            for j in lines:
                img1, img2, label = dataset[j]
                response = agent.eval([img1, img2])
                print("Response: ", response)
                outputs.append((j, response))
                # break

            output_path = os.path.join(prompt_dir, f"{args.label}_return_result={args.return_result}_{args.pretrained}_{args.dataset}_{args.model_name}.txt")
            with open(output_path, "w") as f:
                for o in outputs:
                    f.write(f"{o[0]}\t{o[1]}\n")





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lvlm_pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--lvlm_model_name", type=str, default="llava_qwen")
    parser.add_argument("--eval_lvlm_pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--eval_lvlm_model_name", type=str, default="llava_qwen")
    parser.add_argument("--img1_path", type=str)
    parser.add_argument("--img2_path", type=str)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--split_path", type=str)
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--dataset", type=str, default="lfw")



    args = parser.parse_args()

    main(args)
