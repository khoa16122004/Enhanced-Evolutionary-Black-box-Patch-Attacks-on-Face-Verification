from text_to_text_services import LlamaService
from get_architech import init_lvlm_model
from PIL import Image

llm = LlamaService(model_name="Llama-7b")
lvlm_model, lvlm_image_token, lvlm_special_token = init_lvlm_model(
    "llava-next-interleave-7b", "llava_qwen"
)

img_files = [
    Image.open("../lfw_dataset/lfw_crop_margin_5/Abdoulaye_Wade/Abdoulaye_Wade_0001.jpg").convert("RGB"),
    Image.open("../lfw_dataset/lfw_crop_margin_5/Zhu_Rongji/Zhu_Rongji_0001.jpg").convert("RGB")
]

initial_question = "What is the gender of the person in this image?"

llm_system_prompt = """
You are a detective playing a guessing game. Two independent Vision AI agents each see only one image. Your job is to figure out whether the two images show the SAME PERSON or DIFFERENT PEOPLE.

Rules:
- Ask questions about a **single visual trait** that both agents can describe separately (e.g., hair color, age, emotion).
- DO NOT ask comparative questions (e.g., "Do they look alike?") — the agents only know their own image.
- Each round, you’ll see both agents’ answers to your question. Then you must ask the next strategic question.
- Once you think you have enough clues, respond with "None" to end the questioning and give your final verdict.

Your goal: Solve the case in as few questions as possible.

Now, based on the investigation history so far, what is your next specific visual question for both agents to answer?
Only output one question or the word "None".
"""

llm_prompt_template = "History of investigation:\n{history}\n\nWhat is your next detective question?"

history = []
question = initial_question
max_rounds = 10

for round_idx in range(max_rounds):
    answer_1 = lvlm_model.inference(
        question + lvlm_image_token,
        [img_files[0]],
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8,
        reload=False
    )[0]

    answer_2 = lvlm_model.inference(
        question + lvlm_image_token,
        [img_files[1]],
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8,
        reload=False
    )[0]

    history.append((question, answer_1, answer_2))

    formatted_history = ""
    for q, a1, a2 in history:
        formatted_history += f"Q: {q}\nImage 1 Answer: {a1}\nImage 2 Answer: {a2}\n\n"

    next_question = llm.text_to_text(
        llm_system_prompt,
        llm_prompt_template.format(history=formatted_history)
    )[0].strip()

    print(f"\n🎮 Round {round_idx + 1}")
    print(f"🕵️ Question: {question}")
    print(f"🖼️ Image 1: {answer_1}")
    print(f"🖼️ Image 2: {answer_2}")
    print(f"🧠 Next Question: {next_question}")

    if next_question.lower() == "none":
        print("\n✅ Detective has enough evidence.")

        final_summary_prompt = f"""
Investigation Summary:
{formatted_history}

Now, provide your final analysis.

1. FINAL GUESS: Are these the SAME PERSON or DIFFERENT PEOPLE?
2. DETECTIVE REASONING: Explain how you came to this conclusion.
3. CONFIDENCE LEVEL: High / Medium / Low
4. KEY CLUES: What details were most important?
"""
        final_summary = llm.text_to_text("", final_summary_prompt)[0]
        print("\n🔍 FINAL VERDICT:")
        print(final_summary)
        break

    question = next_question
    input("Press Enter to continue to the next round...")

print("\n🏁 Game Completed.")
print(f"🕵️ Total Questions Asked: {len(history)}")
