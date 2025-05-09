from get_architech import init_lvlm_model
import torch
from PIL import Image
import os
from dataset import get_dataset

class AgentWithDetailedQuestions:
    def __init__(self, lvlm, lvlm_image_token):
        self.lvlm = lvlm
        self.lvlm_image_token = lvlm_image_token

    def ask_question(self, img_files, question, num_samples=10, temperature=0.8):
        # Build prompt for the question
        prompt = f"{question}\nImages: {self.lvlm_image_token * 2}"
        outputs = self.lvlm.inference(
            prompt, img_files,
            num_return_sequences=num_samples, do_sample=True,
            temperature=temperature, reload=False
        )
        
        print("Generated responses: ", len(outputs))
        input("Wait")
        
        
        # Aggregate outputs
        combined_responses = ""
        for i, output in enumerate(outputs):
            combined_responses += f"Response {i+1}:\n{output}\n\n"
        return combined_responses

    def eval(self, img_files, num_samples=10, temperature=0.8):
        questions = [
            "Do the eyes of the two individuals have similar size and shape?",
            "Is there a noticeable difference in the nose length and width between the two individuals?",
            "Are the mouths of the two individuals similar in terms of lip thickness and symmetry?",
            "Do the facial structures, such as the jaw and chin, appear similar?",
            "Do the individuals have similar eyebrow shapes, density, or gaps between brows?"
        ]
        
        all_responses = ""
        for question in questions:
            all_responses += self.ask_question(img_files, question, num_samples, temperature)
        
        conclusion_prompt = (
            "Based on the answers to the following questions and the provied images, determine if the two individuals are the same person:\n"
            f"{all_responses}\n"
            "Return only one word: **same** or **different**."
        )
        
        print("Conclusion prompt: ", conclusion_prompt)
        
        final_decision = self.lvlm.inference(
            conclusion_prompt + self.lvlm_image_token * 2,
            img_files, num_return_sequences=1,
            do_sample=False, temperature=0, reload=False
        )

        return final_decision[0]


def main_with_detailed_questions(args):
    dataset = get_dataset(args.dataset)
    lvlm_model, lvlm_image_token, lvlm_special_token = init_lvlm_model(args.lvlm_pretrained, args.lvlm_model_name)
    
    with torch.no_grad():
        prompt_dir = os.path.join("test_split", 'question')
        os.makedirs(prompt_dir, exist_ok=True)

        img1 = Image.open(args.img1_path).convert("RGB")
        img2 = Image.open(args.img2_path).convert("RGB")

        agent = AgentWithDetailedQuestions(lvlm_model, lvlm_image_token)            
        response = agent.eval([img1, img2], args.num_samples)
        print("Response: ", response)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lvlm_pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--lvlm_model_name", type=str, default="llava_qwen")
    parser.add_argument("--img1_path", type=str)
    parser.add_argument("--img2_path", type=str)
    parser.add_argument("--dataset", type=str, default="lfw")
    parser.add_argument("--num_samples", type=int, default=10)

    args = parser.parse_args()

    main_with_detailed_questions(args)
