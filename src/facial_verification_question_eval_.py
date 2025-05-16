from get_architech import init_lvlm_model
import torch
from PIL import Image
import os
from dataset import get_dataset
import dotenv
from openai import OpenAI
from typing import Optional
import base64
from typing import List
from qwen.model import QWENModel

dotenv.load_dotenv()
class GPTService:
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize the GPTService with a model name and API key.
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in the environment variable 'OPENAI_KEY'.")
        self.client = OpenAI(api_key=self.api_key)

    def text_to_text(self, prompt: str, system_prompt: str) -> str:
        """
        Perform a text-to-text API call.
        """
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.output_text.strip()
        except Exception as e:
            print(f"Error during API call: {e}")
            return "Error occurred during API call."

    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:  
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def image_to_text(self, prompt: str, image_paths: List[str], system_prompt: str) -> str:
        """
        Perform an image-to-text API call using base64-encoded images.
        """
        try:
            base64_images = [self.encode_image(image_path) for image_path in image_paths]
            input_images = [
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}
                for b64 in base64_images
            ]

            response = self.client.responses.create(
                model=self.model_name,
                input=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}] + input_images
                    }
                ]
            )
            return response.output_text.strip()
        except Exception as e:
            print(f"Error during API call: {e}")
            return "Error occurred during API call."

class AgentWithDetailedQuestions:
    def __init__(self, lvlm, lvlm_image_token, llm):
        self.lvlm = lvlm
        self.lvlm_image_token = lvlm_image_token
        self.llm = llm

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
        selection_responses = []
        for i, question in enumerate(questions):
            outputs = self.ask_question(img_files, question, num_samples, temperature)
            all_responses.append(outputs)        
        
            # response_selecion
            selection_voting = f"You will receive a list of responses to a binary question. Your task is to synthesize a final answer based on the ideas that appear most frequently across the responses in the short ways."
            prompt = f"Question: {question}\n Responses: {outputs}\n"
            torch.cuda.empty_cache()
            selection_response = self.llm.text_to_text(prompt, selection_voting)
            torch.cuda.empty_cache()
            selection_responses.append(selection_response)
            
            



        conclusion_prompt = (
          "Given the responses describing facial features in two images, treat each response as a 'vote' indicating whether the images depict the same person or different individuals."
          "Assign greater weight to responses that mention differences in key biometric features (e.g., eye shape, jawline, nose structure)."
          "Based on the overall weighted vote, determine whether the images likely show the same person or not."
          "Here are the responses:"
          f"{selection_responses}"
        )
        self.lvlm.reload()
        final_decision = self.lvlm.inference(
            conclusion_prompt + self.lvlm_image_token * 2,
            img_files, num_return_sequences=1,
            do_sample=True, temperature=0.8, reload=False
        )

        return final_decision[0], all_responses, selection_responses


def main_with_detailed_questions(args):
    dataset = get_dataset(args.dataset)
    lvlm_model, lvlm_image_token, lvlm_special_token = init_lvlm_model(args.lvlm_pretrained, args.lvlm_model_name)
    
    if args.extract_llm == "qwen":
        llm = QWENModel()
    elif args.extract_llm == "gpt4o":
        llm = GPTService(model_name="gpt-4o")
    
    agent = AgentWithDetailedQuestions(lvlm_model, lvlm_image_token, llm)
    output_dir = f"question_pretrained={args.lvlm_pretrained}_modelname={args.lvlm_model_name}_dataset={args.dataset}_num_samples={args.num_samples}_llm={args.extract_llm}"
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
                
            final_decision, all_responses, selection_responses = agent.eval([img1, img2], args.num_samples)
            # print("Final Decision: ", final_decision)
            # print("All Responses: ", all_responses)
            # print("Selection Decision: ", selection_responses)
            # break
            with open(os.path.join(index_dir, "decide.txt"), "w") as f:
                f.write(f"{final_decision}\n")
            for j, response in enumerate(selection_responses):
                with open(os.path.join(index_dir, f"selection_{j}.txt"), "w") as f:  
                    f.write(f"{response}")
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
    parser.add_argument("--extract_llm", type=str, default="qwen")

    args = parser.parse_args()

    main_with_detailed_questions(args)
