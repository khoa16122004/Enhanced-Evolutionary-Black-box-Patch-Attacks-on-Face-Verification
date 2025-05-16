from get_architech import init_lvlm_model
import torch
from PIL import Image
import os
from dataset import get_dataset

class AgentWithDetailedQuestions:
    def __init__(self, lvlm, lvlm_image_token):
        self.lvlm = lvlm
        self.lvlm_image_token = lvlm_image_token

    def ask_question(self, img_files, question, num_samples=1, temperature=0.8):
        # Build prompt for the question
        prompt = f"{question}\nImages: {self.lvlm_image_token * 2}"
        outputs = self.lvlm.inference(
            prompt, img_files,
            num_return_sequences=num_samples, do_sample=True,
            temperature=temperature, reload=False
        )

        return outputs

    def eval(self, img_files, num_samples=1, temperature=0.8):
        questions = [
            "Do the eyes of the two individuals have similar size and shape?",
            "Is there a noticeable difference in the nose length and width between the two individuals?",
            "Are the mouths of the two individuals similar in terms of lip thickness and symmetry?",
            "Do the facial structures, such as the jaw and chin, appear similar?",
            "Do the individuals have similar eyebrow shapes, density, or gaps between brows?"
        ]
        
        
        all_responses = []
        for i, question in enumerate(questions):
            outputs = self.ask_question(img_files, question, num_samples, temperature)
            all_responses.append(outputs)    
                
        conclusion_prompt = (
            "Given the responses to the facial biometric questions and the provided images, determine whether the two individuals are the same person:\n"
            f"{all_responses}\n"
            "Give more weight to responses indicating differences in features.\n"
            "Return only one word: **same** or **different**."
        )
        
        final_decision = self.lvlm.inference(
            conclusion_prompt + self.lvlm_image_token * 2,
            img_files, num_return_sequences=1,
            do_sample=False, temperature=0.0, reload=False
        )

        return final_decision[0], all_responses


def main_with_detailed_questions(args):
    dataset = get_dataset(args.dataset)
    lvlm_model, lvlm_image_token, lvlm_special_token = init_lvlm_model(args.lvlm_pretrained, args.lvlm_model_name)
    
    agent = AgentWithDetailedQuestions(lvlm_model, lvlm_image_token)
    output_dir = f"question_pretrained={args.lvlm_pretrained}_modelname={args.lvlm_model_name}_dataset={args.dataset}_num_samples={args.num_samples}"
    os.makedirs(output_dir, exist_ok=True)
    
    
    num_0 = 0
    num_1 = 0
    with torch.no_grad():
        for i in range(310, len(dataset)):    
            
            if num_0 > 10 and num_1 > 10:
                break

            
           
            
            img1, img2, label = dataset[i]
            if label == 0:
                num_0 += 1
                if num_0 > 10:
                    continue
                else:
                    index_dir = os.path.join(output_dir, str(i))
                    os.makedirs(index_dir, exist_ok=True)
            else:
                num_1 += 1
                if num_1 > 10:
                    continue
                else:
                    index_dir = os.path.join(output_dir, str(i))
                    os.makedirs(index_dir, exist_ok=True)
                
            final_decision, all_responses = agent.eval([img1, img2], args.num_samples)
            
            with open(os.path.join(index_dir, "decide.txt"), "w") as f:
                f.write(f"{final_decision}\n")
            
            for j, question in enumerate(all_responses):
                question_dir = os.path.join(index_dir, f"question_{j}")
                os.makedirs(question_dir, exist_ok=True)
                
                for k, response in enumerate(all_responses[j]):
                    response_path = os.path.join(question_dir, f"response_{k}.txt")
                    with open(response_path, "w") as f:
                        f.write(f"{response}\n")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lvlm_pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--lvlm_model_name", type=str, default="llava_qwen")
    parser.add_argument("--dataset", type=str, default="lfw")
    parser.add_argument("--num_samples", type=int, default=10)

    args = parser.parse_args()

    main_with_detailed_questions(args)
