from text_to_text_services import LlamaService
from get_architech import init_lvlm_model
from PIL import Image

# Khởi tạo mô hình
llm = LlamaService(model_name="Llama-7b")
lvlm_model, lvlm_image_token, _ = init_lvlm_model("llava-next-interleave-7b", "llava_qwen")

# Tải ảnh
img1 = Image.open("../lfw_dataset/lfw_original/Adel_Al-Jubeir/Adel_Al-Jubeir_0001.jpg").convert("RGB")
img2 = Image.open("../lfw_dataset/lfw_original/Ziwang_Xu/Ziwang_Xu_0001.jpg").convert("RGB")
img1.save("test1.png")
img2.save("test2.png")

# Thiết lập trò chơi
initial_question = "Describe the person in the image."
llm_system_prompt = """
🎯 GAME INSTRUCTIONS

ROLE:
You are a detective playing a question-answer game inspired by "20 Questions".

SETUP:
- Two witnesses are each given one image of a person.
- Each witness can only describe the person in their own image.
- They do not see each other’s image and cannot guess or infer about the other image.

RULES:
- In each turn, you must ask **a single question** to both witnesses.
- The question should help you extract distinguishing details (e.g., facial features, age, gender, accessories).
- Do **not** ask person-specific comparisons (e.g., “Does person 1 have X and person 2 not?”).
- Ask general questions that both witnesses can answer independently.

OBJECTIVE:
- Your goal is to collect enough information to determine whether the two images are of the **same person** or **different people**.
- After reviewing the Q&A history, if you believe you have enough information, return `None`.

RESPONSE FORMAT:
Only output the next best question to ask. If enough information has been gathered, output exactly:
None
"""


llm_prompt_template = "History:\n{history}"

history = []
question = initial_question
max_rounds = 10
should_stop = False

for round_idx in range(max_rounds):
    full_question = f"You can't refuse to answer the question about the facial image\nquestion: {question}\nimage:{lvlm_image_token}"
    answer_1 = lvlm_model.inference(full_question, [img1], num_return_sequences=1, do_sample=True, temperature=0.8, reload=False)[0]
    answer_2 = lvlm_model.inference(full_question, [img2], num_return_sequences=1, do_sample=True, temperature=0.8, reload=False)[0]

    # Cập nhật lịch sử
    history.append((question, answer_1, answer_2))

    # Hiển thị
    print(f"\n🎮 Game Round {round_idx + 1}")
    print(f"🕵️ Detective Question: {question}")
    print(f"👤 Witness #1: {answer_1}")
    print(f"👤 Witness #2: {answer_2}")

    # Định dạng lại lịch sử
    formatted_history = "\n".join(
        f"Q: {q}\nWitness 1: {a1}\nWitness 2: {a2}" for q, a1, a2 in history
    )

    # Dự đoán câu hỏi tiếp theo
    next_question = llm.text_to_text(
        llm_system_prompt,
        llm_prompt_template.format(history=formatted_history)
    )[0]

    print(f"🤔 Detective's Next Strategy: {next_question}")

    if "none" in next_question.lower():
        should_stop = True
        break

    question = next_question
    input("Press Enter to continue to the next round...")

# Tổng kết khi kết thúc vòng lặp hoặc nhận được "None"
final_summary_prompt = f"""
The investigation is complete! You've been asking questions to two witnesses, each holding a different image.

Here's your complete investigation history:
{formatted_history}

Now provide your final verdict:

🎯 FINAL GUESS: Are these the SAME PERSON or DIFFERENT PEOPLE?

🔍 DETECTIVE REASONING: What key evidence led to your conclusion? Explain your logical deduction process.

📊 CONFIDENCE LEVEL: How confident are you? (High/Medium/Low)

🎮 GAME SUMMARY: Briefly summarize the most important clues that solved the case.
"""

final_summary = llm.text_to_text("", final_summary_prompt)[0]

print("\n🏁 GAME OVER!")
print(f"🔢 Total Investigation Rounds: {len(history)}")
print("\n🏆 DETECTIVE'S FINAL VERDICT:")
print(final_summary)
