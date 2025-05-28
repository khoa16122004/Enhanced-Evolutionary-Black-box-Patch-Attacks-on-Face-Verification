from text_to_text_services import LlamaService
from get_architech import init_lvlm_model
from PIL import Image

llm = LlamaService(model_name="Llama-7b")
lvlm_model, lvlm_image_token, lvlm_special_token = init_lvlm_model("llava-next-interleave-7b", 
                                                                   "llava_qwen")

img_files = [
    Image.open("../lfw_dataset/lfw_crop_margin_5/Adel_Al-Jubeir/Adel_Al-Jubeir_0001.jpg"),
    Image.open("../lfw_dataset/lfw_crop_margin_5/Ziwang_Xu/Ziwang_Xu_0001.jpg")
]

initial_question = "Let's start the guessing game! What is the gender of the person in this image?"

llm_system_prompt = """
You are playing a guessing game! Your goal is to determine if two facial images show the same person or different people.

Here's the twist - you CANNOT see the images yourself. You can only ask questions, and two Vision AI assistants will look at each image separately and answer your questions.

Game Rules:
- Ask strategic questions about facial features, appearance, or characteristics
- You'll receive two separate answers (one for each image)
- Use the answers to guide your next question
- Keep asking until you're confident about your guess
- When you have enough information to make a decision, respond with "None"

Think like a detective! Look for similarities and differences in the answers. Ask about:
- Physical features (eyes, nose, mouth, face shape)
- Age, gender, ethnicity
- Distinctive marks, accessories, expressions
- Hair style, facial hair
- Any unique characteristics

Based on the Q&A history, what's your next strategic question to solve this mystery?
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
        formatted_history += f"Q: {q}\nImage 1 Answer: {a1}\nImage 2 Answer: {a2}\n\n"

    # Generate next question based on both answers
    next_question = llm.text_to_text(llm_system_prompt, llm_prompt_template.format(history=formatted_history))[0]

    print(f"\n🎮 Game Round {round_idx + 1}")
    print(f"🕵️ Detective Question: {question}")
    print(f"👤 Vision AI #1 (Image 1): {answer_1}")
    print(f"👤 Vision AI #2 (Image 2): {answer_2}")
    print(f"🤔 Detective's Next Strategy: {next_question}")

    if "None" in next_question or "none" in next_question.lower():
        print("\n🎯 GAME OVER! Detective has reached a conclusion!")
        
        # Final summary with all Q&A pairs
        final_summary_prompt = f"""
The guessing game is over! As the detective, you've been asking questions to two Vision AI assistants about two different images.

Here's your complete investigation history:
{formatted_history}

Now it's time for your final verdict! Please provide:

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