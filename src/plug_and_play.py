from text_to_text_services import LlamaService
from get_architech import init_lvlm_model
from PIL import Image

llm = LlamaService(model_name="Llama-7b")
lvlm_model, lvlm_image_token, lvlm_special_token = init_lvlm_model("llava-next-interleave-7b", 
                                                                   "llava_qwen")

img_files = [
    Image.open("../lfw_dataset/lfw_original/Adel_Al-Jubeir/Adel_Al-Jubeir_0001.jpg").convert("RGB"),
    Image.open("../lfw_dataset/lfw_original/Ziwang_Xu/Ziwang_Xu_0001.jpg").convert("RGB")
]

img_files[0].save("test1.png")
img_files[1].save("test2.png")

initial_question = "Describe the person in the image."

llm_system_prompt = """
🎮 DETECTIVE CHALLENGE: Guess if two faces are the same person using the FEWEST questions possible!

🕵️ Your Mission: You're a master detective who cannot see the images. Two witnesses each have one different image, and you'll ask them the same question. They don't know what the other person's image looks like.

🎯 GAME RULES:
- Ask ONE question that will be given to both witnesses
- Each witness will describe only their own image using the same question
- Each question costs points - fewer questions = higher score!
- Compare the two answers yourself to find similarities/differences
- When you're confident about your conclusion, respond with "None"
- Frame questions so they work for any single image

⚠️ IMPORTANT: 
- Write questions like "Describe the [feature]" or "What is the [aspect]?"
- DON'T ask comparative questions since each person only sees one image
- Your question will be asked to two different people with two different images
- Make questions specific enough to get detailed, comparable answers

⚡ Only return your next strategic question. Nothing else. If you have enough evidence, return "None".

What's your next detective question?
"""

llm_prompt_template = "History:\n{history}"

history = []
question = initial_question
max_rounds = 10

for round_idx in range(max_rounds):
    # Ask the question to the first image
    answer_1 = lvlm_model.inference(
        question + lvlm_image_token,
        [img_files[0]],  # Pass as list for consistency
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8,
        reload=False
    )[0]
    
    # Ask the same question to the second image  
    answer_2 = lvlm_model.inference(
        question + lvlm_image_token,
        [img_files[1]],  # Pass as list for consistency
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8,
        reload=False
    )[0]

    # Store both answers in history
    history.append((question, answer_1, answer_2))

    # Format history for LLM - include both answers for comparison
    formatted_history = ""
    for q, a1, a2 in history:
        formatted_history += f"Q: {q}\nWitness 1: {a1}\nWitness 2: {a2}\n\n"

    # Generate next question based on both answers
    next_question = llm.text_to_text(llm_system_prompt, llm_prompt_template.format(history=formatted_history))[0]

    print(f"\n🎮 Game Round {round_idx + 1}")
    print(f"🕵️ Detective Question: {question}")
    print(f"👤 Witness #1: {answer_1}")
    print(f"👤 Witness #2: {answer_2}")
    print(f"🤔 Detective's Next Strategy: {next_question}")

    if "None" in next_question or "none" in next_question.lower():
        print("\n🎯 GAME OVER! Detective has reached a conclusion!")
        
        # Final summary with all Q&A pairs
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
        print("\n🏆 DETECTIVE'S FINAL VERDICT:")
        print(final_summary)
        break

    question = next_question
    input("Press Enter to continue to the next round...")

print("\n🎮 GAME COMPLETED!")
print(f"🔢 Total Investigation Rounds: {len(history)}")
print(f"🕵️ Questions Asked: {[q for q, _, _ in history]}")