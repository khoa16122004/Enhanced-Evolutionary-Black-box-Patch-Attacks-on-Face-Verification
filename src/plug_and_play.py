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

FOCUS ONLY ON:
- Face shape, eye shape/color, nose structure
- Skin tone, facial hair, distinctive marks
- Age appearance, hair color/style

ASK ONE QUESTION PER TURN. If you have enough biometric info to decide, return: STOP

Next question:"""

# Chọn chế độ chơi
print("🔍 FACE VERIFICATION GAME")
print("=" * 40)
print("Choose game mode:")
print("1. AUTO - LLM asks questions automatically")
print("2. MANUAL - You ask questions manually")

mode = input("Enter mode (1 or 2): ").strip()
is_manual_mode = mode == "2"

# Thiết lập ban đầu
initial_question = "Describe the person's facial features in detail."
history = []
question = initial_question
max_rounds = 4

print(f"\n🎮 Mode: {'MANUAL' if is_manual_mode else 'AUTO'}")
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

    # Tạo context ngắn cho LLM (chỉ cho chế độ AUTO)
    if not is_manual_mode:
        context = "\n---\n".join(history)
        
        # Hỏi câu hỏi tiếp theo
        next_question = llm.text_to_text(llm_system_prompt, f"History:\n{context}")[0].strip()
    
    if is_manual_mode:
        # Chế độ thủ công - người chơi nhập câu hỏi
        print("\n🎯 Your turn to ask a question!")
        print("Focus on facial biometrics: eyes, nose, face shape, skin tone, hair, etc.")
        print("Type 'STOP' when you have enough information to make a decision.")
        
        next_question = input("Your question: ").strip()
        
        if next_question.upper() == "STOP":
            break
            
        question = next_question
    else:
        # Chế độ tự động - LLM tự hỏi
        print(f"🤔 Next: {next_question}")

        if "STOP" in next_question.upper():
            break

        question = next_question
        input("\nPress Enter for next round...")

# Kết luận cuối game - tập trung vào biometrics
if is_manual_mode:
    print("\n🎯 Time for your final decision!")
    print("Based on the evidence you've gathered:")
    final_decision = input("Your verdict (SAME/DIFFERENT): ").strip()
    reasoning = input("Your reasoning (key biometric evidence): ").strip()
    confidence = input("Your confidence (HIGH/MEDIUM/LOW): ").strip()
    
    print("\n" + "=" * 40)
    print("🏆 YOUR FINAL ANALYSIS")
    print("=" * 40)
    print(f"VERDICT: {final_decision}")
    print(f"REASONING: {reasoning}")
    print(f"CONFIDENCE: {confidence}")
else:
    final_prompt = f"""You are a detective completing a face verification investigation. 

Your complete investigation record:
{chr(10).join(history)}

Now summarize your findings and give ONE clear final answer: Are these the SAME PERSON or DIFFERENT PEOPLE?

Base your conclusion on the facial biometric evidence only. Be consistent with the evidence you've gathered.

Final verdict:"""

    final_verdict = llm.text_to_text("", final_prompt)[0]

    print("\n" + "=" * 40)
    print("🏆 DETECTIVE'S FINAL VERDICT")
    print("=" * 40)
    print(final_verdict)
print(f"\n📊 Rounds completed: {len(history)}")