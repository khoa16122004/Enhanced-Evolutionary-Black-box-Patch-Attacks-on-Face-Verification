from get_architech import init_lvlm_model
import torch
from PIL import Image

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
            "You are given two facial images. Carefully compare them in the following biometric regions:\n"
            "- Eyes: shape, distance, eyelids\n"
            "- Nose: size, shape, nostrils\n"
            "- Mouth: width, lip shape, corners\n"
            "- Jawline and face shape: contour, chin\n"
            "- Eyebrows: shape, thickness, distance\n"
            "\nBased on these comparisons, decide whether the two images likely depict the same person. "
            "Provide a detailed explanation before making your conclusion.\n\nImages:"
        )

        full_prompt = prompt_base + self.lvlm_image_token * 2
        outputs = self.lvlm.inference(full_prompt, img_files, num_return_sequences=10, do_sample=True, 
                                     temperature=temperature, reload=False)
        
        print(len(outputs))
        for output in outputs:
            print("Response: ", output)
        # previous_output = ""  

        # for i in range(self.steps):
        #     prompt = prompt_base + self.lvlm_image_token * 2 + "\n\nPrevious attempt:\n" + previous_output
            
        #     output = self.lvlm.inference(prompt, img_files, temperature=temperature, reload=False)[0].replace("\n", "")
            
        #     final_prompt = (
        #         "You will be given two facial images and a description. "
        #         "Decide whether the explanation provided is convincing enough and sufficient to conclude the identity. "
        #         "If yes, return 'True', otherwise return 'False'.\n\n"
        #         f"Description: \n{output}\n Images: "
        #     )


        #     eval_output = self.eval_lvlm.inference(final_prompt + self.eval_lvlm_image_token * 2, 
        #                                            img_files, 
        #                                            temperature=temperature, reload=False)[0].replace("\n", "")

        #     print(f"Step {i}: Large VLM output: {output}, Eval output: {eval_output}\n")

        #     previous_output = f"Step {i}: {output}, Eval output: {eval_output}\n"

        #     if eval_output == "True":
        #         return output

        return None

def main(args):
    lvlm_model, lvlm_image_token, lvlm_special_token = init_lvlm_model(args.lvlm_pretrained, args.lvlm_model_name)
    eval_lvlm, eval_lvlm_image_token, eval_lvlm_special_token = init_lvlm_model(args.eval_lvlm_pretrained, args.eval_lvlm_model_name)
    
    with torch.no_grad():
        img1 = Image.open(args.img1_path).convert("RGB")
        img2 = Image.open(args.img2_path).convert("RGB")

        agent = Agent(lvlm_model, lvlm_image_token,
                      eval_lvlm, eval_lvlm_image_token,
                      args.steps)

        output = agent.eval([img1, img2])
        print(f"Final output: {output}")

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
    args = parser.parse_args()

    main(args)
