from get_architech import init_lvlm_model
import torch
import argparse
from PIL import Image
from tqdm import tqdm

class Agent:
    def __init__(self, lvlm, eval_lvlm, steps):
        self.lvlm = lvlm
        self.steps = steps
        self.eval_lvlm = eval_lvlm
                
    def eval(self, img_files, temperature=0.5):
        prompt_base = (
            "You are given two facial images. Compare them in terms of biometric regions. "
            "Then decide whether they depict the same person."
        )
        for i in range(self.steps):
            output = self.lvlm.inference(prompt_base, img_files, 
                                         temperature=temperature, reload=False)[0].replace("\n", "")
            
            
            final_prompt = (
                "Given the description below about two facial images, "
                "decide whether it is a correct and sufficient explanation to conclude identity. "
                "If yes, return 'True', otherwise return 'False'.\n\n"
                f"Description: \n{output}"
            )
            eval_output = self.eval_lvlm.inference(final_prompt, img_files, 
                                                   temperature=temperature, reload=False)[0].replace("\n", "")
            print(f"Step {i}: Large VLM output: {output}, Eval output: {eval_output}\n")

            
            if eval_output == "True":
                return output
        return None

def main(args):
    lvlm = init_lvlm_model(args.lvlm_pretrained, args.lvlm_model_name)
    eval_lvlm = init_lvlm_model(args.eval_lvlm_pretrained, args.eval_lvlm_model_name)
    with torch.no_grad():
        img1 = Image.open(args.img1_path).convert("RGB")
        img2 = Image.open(args.img2_path).convert("RGB")


        
        agent = Agent(lvlm, eval_lvlm, args.steps)
        
        # lvlm.reload()
        # eval_lvlm.reload()
        
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
    
     