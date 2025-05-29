from text_to_text_services import LlamaService
from get_architech import init_lvlm_model
from PIL import Image

# Khởi tạo mô hình
llm = LlamaService(model_name="Llama-7b")
lvlm_model, lvlm_image_token, _ = init_lvlm_model("llava-next-interleave-7b", "llava_qwen")

# Tải ảnh
img1 = Image.open("../lfw_dataset/lfw_original/Ziwang_Xu/Ziwang_Xu_0001.jpg").convert("RGB")
img2 = Image.open("../lfw_dataset/lfw_original/Ziwang_Xu/Ziwang_Xu_0001.jpg").convert("RGB")
img1.save("test1.png")
img2.save("test2.png")

# System prompt đơn giản hóa
llm_system_prompt = """You are analyzing facial features to verify if two people are the same.

TASK: Ask strategic questions about facial biometrics to determine if two images show the same person.


ASK ONE QUESTION PER TURN. If you have enough biometric info to decide, return: STOP

Next question:"""

# Thiết lập ban đầu
initial_question = "Describe the person's facial features in detail."
history = []
question = initial_question
max_rounds = 4

print("🔍 FACE VERIFICATION GAME")
print("=" * 40)

for round_idx in range(max_rounds):
    # Hỏi cả hai witness
    full_question = f"Focus on facial features only.\nQuestion: {question}\nImage: {lvlm_image_token}"
    
    answer_1 = lvlm_model.inference(full_question, [img1], num_return_sequences=1, 
                                  do_sample=True, temperature=0.7, reload=False)[0]
    answer_2 = lvlm_model.inference(full_question, [img2], num_return_sequences=1, 
                                  do_sample=True, temperature=0.7, reload=False)[0]

    # Lưu lịch sử ngắn gọn
    history.append(f"Q: {question}\nA1: {answer_1}\nA2: {answer_2}")

    print(f"\nRound {round_idx + 1}:")
    print(f"❓ {question}")
    print(f"👤 Image 1: {answer_1}")
    print(f"👤 Image 2: {answer_2}")

    # Tạo context ngắn cho LLM
    context = "\n---\n".join(history)
    
    # Hỏi câu hỏi tiếp theo
    next_question = llm.text_to_text(llm_system_prompt, f"History:\n{context}")[0].strip()
    
    print(f"🤔 Next: {next_question}")

    if "STOP" in next_question.upper():
        break

    question = next_question
    input("\nPress Enter for next round...")

# # Kết luận cuối game - tập trung vào biometrics
# final_prompt = f"""Based on facial biometric analysis:

# EVIDENCE:
# {chr(10).join(history)}

# INSTRUCTIONS:
# 1. Compare ONLY facial biometric features (eyes, nose, face shape, skin tone, etc.)
# 2. Ignore clothing, background, image quality
# 3. Give final verdict: SAME PERSON or DIFFERENT PEOPLE
# 4. List 3 key biometric evidence points
# 5. Rate confidence: HIGH/MEDIUM/LOW

# VERDICT:"""

# final_verdict = llm.text_to_text("", final_prompt)[0]

# print("\n" + "=" * 40)
# print("🏆 FINAL ANALYSIS")
# print("=" * 40)
# print(final_verdict)
# print(f"\n📊 Rounds completed: {len(history)}")