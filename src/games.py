from get_architech import init_lvlm_model
import torch
import os
import json
from dataset import get_dataset
from text_to_text_services import QwenService, GPTService, LlamaService
from tqdm import tqdm
from PIL import Image
import time

class FaceVerificationGame:
    def __init__(self, lvlm_model, lvlm_image_token, llm, max_rounds=4):
        self.lvlm_model = lvlm_model
        self.lvlm_image_token = lvlm_image_token
        self.llm = llm
        self.max_rounds = max_rounds
        
        self.system_prompt = """You are analyzing facial features to verify if two people are the same.

TASK: Ask strategic questions about facial biometrics to determine if two images show the same person.

FOCUS ONLY ON:
- Face shape, eye shape/color, nose structure
- Skin tone, facial hair, distinctive marks
- Age appearance, hair color/style

ASK ONE QUESTION PER TURN. If you have enough biometric info to decide OR if nearing max rounds, return: STOP

Next question:"""

    def play_auto_mode(self, img1, img2, save_images=True, image_dir=None):
        """
        Chạy game ở chế độ AUTO - LLM tự động hỏi câu hỏi
        """
        history = []
        question = "Describe the person's facial features in detail."
        
        # Lưu ảnh nếu cần
        if save_images and image_dir:
            img1.save(os.path.join(image_dir, "image1.png"))
            img2.save(os.path.join(image_dir, "image2.png"))
        
        game_log = {
            "mode": "AUTO",
            "max_rounds": self.max_rounds,
            "start_time": time.time(),
            "rounds": [],
            "final_verdict": None,
            "total_rounds": 0
        }
        
        for round_idx in range(self.max_rounds):
            current_round = round_idx + 1
            
            # Hỏi cả hai witness
            full_question = f"Focus on facial features only.\nQuestion: {question}\nImage: {self.lvlm_image_token}"
            
            try:
                answer_1 = self.lvlm_model.inference(full_question, [img1], num_return_sequences=1, 
                                                   do_sample=True, temperature=0.7, reload=False)[0]
                answer_2 = self.lvlm_model.inference(full_question, [img2], num_return_sequences=1, 
                                                   do_sample=True, temperature=0.7, reload=False)[0]
            except Exception as e:
                game_log["error"] = str(e)
                break
            
            # Lưu round vào log
            round_data = {
                "round": current_round,
                "question": question,
                "answer_1": answer_1,
                "answer_2": answer_2,
                "timestamp": time.time()
            }
            game_log["rounds"].append(round_data)
            
            # Lưu lịch sử với round number
            history.append(f"Round {current_round} - Q: {question}\nA1: {answer_1}\nA2: {answer_2}")
            
            # Kiểm tra xem đã đến round cuối chưa
            if current_round == self.max_rounds:
                break
            
            # Tạo context và hỏi câu hỏi tiếp theo
            context = "\n---\n".join(history)
            llm_prompt = f"Current round: {current_round}/{self.max_rounds}\nHistory:\n{context}"
            
            try:
                next_question = self.llm.text_to_text(self.system_prompt, llm_prompt)[0].strip()
            except Exception as e:
                game_log["llm_error"] = str(e)
                break
            
            if "STOP" in next_question.upper():
                game_log["stopped_early"] = True
                break
            
            question = next_question
        
        # Final verdict
        game_log["total_rounds"] = len(game_log["rounds"])
        
        final_prompt = f"""You are a detective completing a face verification investigation. 

Your complete investigation record:
{chr(10).join(history)}

Now summarize your findings and give ONE clear final answer: Are these the SAME PERSON or DIFFERENT PEOPLE?

Base your conclusion on the facial biometric evidence only. Be consistent with the evidence you've gathered.

Final verdict:"""
        
        try:
            final_verdict = self.llm.text_to_text("", final_prompt)[0]
            game_log["final_verdict"] = final_verdict
        except Exception as e:
            game_log["final_verdict_error"] = str(e)
        
        game_log["end_time"] = time.time()
        game_log["duration"] = game_log["end_time"] - game_log["start_time"]
        
        return game_log

    def play_single_pair(self, images, image_dir=None):
        """
        Chạy game cho một cặp ảnh
        """
        img1, img2 = images
        return self.play_auto_mode(img1, img2, save_images=True, image_dir=image_dir)


def main_with_detailed_questions(args):
    dataset = get_dataset(args.dataset)
    lvlm_model, lvlm_image_token, lvlm_special_token = init_lvlm_model(args.lvlm_pretrained, args.lvlm_model_name)
    
    if args.extract_llm == "qwen":
        llm = QwenService(model_name="qwen")
    elif args.extract_llm == "gpt4o":
        llm = GPTService(model_name="gpt-4o")
    else:
        llm = LlamaService(model_name="Llama-7b")
    
    game = FaceVerificationGame(lvlm_model, lvlm_image_token, llm, args.max_rounds)
    
    # Tạo output directory
    output_dir = f"games_pretrained={args.lvlm_pretrained}_modelname={args.lvlm_model_name}_dataset={args.dataset}_num_samples={args.num_samples}_llm={args.extract_llm}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Counters để stop theo điều kiện
    num_same = 0  # label = 1 (same person)
    num_diff = 0  # label = 0 (different person)
    
    # Metadata file
    metadata = {
        "args": vars(args),
        "total_processed": 0,
        "same_person_pairs": 0,
        "different_person_pairs": 0,
        "results": []
    }
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            # Kiểm tra điều kiện dừng
            if num_same >= args.num_samples and num_diff >= args.num_samples:
                print(f"Completed: {args.num_samples} same pairs and {args.num_samples} different pairs")
                break
            
            img1, img2, label = dataset[i]
            
            # Skip nếu đã đủ số lượng cho loại này
            if label == 0 and num_same >= args.num_samples:
                continue
            if label == 0 and num_diff >= args.num_samples:
                continue
            
            # Tạo thư mục cho sample này
            index_dir = os.path.join(output_dir, f"sample_{i}_label_{label}")
            os.makedirs(index_dir, exist_ok=True)
            
            # Chạy game
            try:
                game_history = game.play_single_pair([img1, img2], index_dir)
                
                # Lưu game history
                with open(os.path.join(index_dir, "game_log.json"), "w", encoding="utf-8") as f:
                    json.dump(game_history, f, indent=2, ensure_ascii=False)
                
                # Cập nhật counters
                if label == 0:
                    num_same += 1
                else:
                    num_diff += 1
                
                # Lưu vào metadata
                result_entry = {
                    "sample_id": i,
                    "true_label": int(label),
                    "rounds_played": game_history["total_rounds"],
                    "final_verdict": game_history.get("final_verdict", "ERROR"),
                    "duration": game_history.get("duration", 0),
                    "stopped_early": game_history.get("stopped_early", False),
                    "has_error": "error" in game_history or "llm_error" in game_history
                }
                metadata["results"].append(result_entry)
                
            except Exception as e:
                print(f"Error processing sample {i}: {str(e)}")
                # Lưu error log
                error_log = {"sample_id": i, "error": str(e), "label": int(label)}
                with open(os.path.join(index_dir, "error_log.json"), "w") as f:
                    json.dump(error_log, f, indent=2)
    
    # Cập nhật metadata cuối cùng
    metadata["total_processed"] = len(metadata["results"])
    metadata["same_person_pairs"] = num_same
    metadata["different_person_pairs"] = num_diff
    
    # Lưu metadata tổng
    with open(os.path.join(output_dir, "experiment_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Experiment completed!")
    print(f"Total processed: {metadata['total_processed']}")
    print(f"Same person pairs: {num_same}")
    print(f"Different person pairs: {num_diff}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lvlm_pretrained", type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument("--lvlm_model_name", type=str, default="llava_qwen")
    parser.add_argument("--dataset", type=str, default="lfw_original")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--extract_llm", type=str, default="Llama-7b")
    parser.add_argument("--max_rounds", type=int, default=4)

    args = parser.parse_args()

    main_with_detailed_questions(args)