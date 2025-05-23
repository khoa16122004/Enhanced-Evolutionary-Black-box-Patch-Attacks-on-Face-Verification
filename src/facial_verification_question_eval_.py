from get_architech import init_lvlm_model
import torch
import os
from dataset import get_dataset
from text_to_text_services import QwenService, GPTService, LlamaService
from tqdm import tqdm

class AgentWithDetailedQuestions:
    def __init__(self, lvlm, lvlm_image_token, llm):
        self.lvlm = lvlm
        self.lvlm_image_token = lvlm_image_token
        self.llm = llm

        self.questions = [
            "Do the eyes of the two individuals have similar size and shape?",
            "Is there a noticeable difference in the nose length and width between the two individuals?",
            "Are the mouths of the two individuals similar in terms of lip thickness and symmetry?",
            "Do the facial structures, such as the jaw and chin, appear similar?",
            "Do the individuals have similar eyebrow shapes, density, or gaps between brows?"
        ]

        self.selection_voting = (
            "You will receive multiple brief opinions on a binary question.",
            "Treat these as votes and determine the majority viewpoint.",
            "Summarize the overall consensus in a short sentence, focusing only on the main idea."
        )
      



        self.conclusion_prompt_template = (
          "Given the responses describing facial features in two images, treat each response as a 'vote' indicating whether the images depict the same person or different individuals."
          "Assign greater weight to responses that mention differences in key biometric features (e.g., eye shape, jawline, nose structure)."
          "Based on the overall weighted vote, determine whether the images likely show the same person or not."
          "Here are the responses:"
          "{responses}"
        )

    def ask_question(self, img_files, question, num_samples=1, temperature=0.8):
        prompt = f"{question}\nImages: {self.lvlm_image_token * 2}"
        print("Prompt ask question: ", prompt)
        outputs = self.lvlm.inference(
            prompt, img_files,
            num_return_sequences=num_samples, do_sample=True,
            temperature=temperature, reload=False
        )

        return outputs

    def eval(self, img_files, num_samples=3, temperature=0.8):
        all_responses = []
        selection_responses = []

        for question in self.questions:
            
            # output from each question
            outputs = self.ask_question(img_files, question, num_samples, temperature)
            print("Outputs: ", outputs)
            all_responses.append(outputs)


            # voting for each question
            prompt = f"Question: {question}\n Responses: {outputs}\n"
            print("Prompt: ", prompt)
            selection_response = self.llm.text_to_text(self.selection_voting, prompt)
            selection_responses.append(selection_response)

        print("selection responses: ", selection_responses)
        conclusion_prompt = self.conclusion_prompt_template.format(responses=selection_responses)
        # self.lvlm.reload()
        final_decision = self.lvlm.inference(
            conclusion_prompt + self.lvlm_image_token * 2,
            img_files, num_return_sequences=1,
            do_sample=True, temperature=0.8, reload=False
        )[0]

        return final_decision, all_responses, selection_responses


def main_with_detailed_questions(args):
    dataset = get_dataset(args.dataset)
    lvlm_model, lvlm_image_token, lvlm_special_token = init_lvlm_model(args.lvlm_pretrained, args.lvlm_model_name)
    
    if args.extract_llm == "qwen":
        llm = QwenService(model_name="qwen")
    elif args.extract_llm == "gpt4o":
        llm = GPTService(model_name="gpt-4o")
    else:
        llm = LlamaService(model_name="Llama-7b")
    
    agent = AgentWithDetailedQuestions(lvlm_model, lvlm_image_token, llm)
    output_dir = f"question_pretrained={args.lvlm_pretrained}_modelname={args.lvlm_model_name}_dataset={args.dataset}_num_samples={args.num_samples}_llm={args.extract_llm}"
    os.makedirs(output_dir, exist_ok=True)
    num_0 = 0
    num_1 = 0
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):    


            img1, img2, label = dataset[i]

            index_dir = os.path.join(output_dir, str(i))
            os.makedirs(index_dir, exist_ok=True)

            index_dir = os.path.join(output_dir, str(i))
            os.makedirs(index_dir, exist_ok=True)
                
            final_decision, all_responses, selection_responses = agent.eval([img1, img2], args.num_samples)


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
    parser.add_argument("--extract_llm", type=str, default="Llama-7b")

    args = parser.parse_args()

    main_with_detailed_questions(args)
