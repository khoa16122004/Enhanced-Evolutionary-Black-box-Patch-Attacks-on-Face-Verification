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

initial_question = "What is the gender?"

llm_system_prompt = """
You are analyzing whether two facial images belong to the same person.
You have received the following Q&A history where each question was asked to both images separately.
Based on the previous answers from both images, please propose a follow-up question that explores the comparison in more detail.
Focus on features that could help determine if these are the same person.
If enough information has been gathered, please respond with "None".
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

    print(f"\n🔁 Round {round_idx + 1}")
    print(f"❓ Question: {question}")
    print(f"✅ Image 1 Answer: {answer_1}")
    print(f"✅ Image 2 Answer: {answer_2}")
    print(f"🤔 Next Question Generated: {next_question}")

    if "None" in next_question or "none" in next_question.lower():
        print("\n🛑 LLM indicates sufficient information gathered.")
        
        # Final summary with all Q&A pairs
        final_summary_prompt = f"""
Based on the following Q&A history, please summarize the reasoning and conclude whether the two faces belong to the same person or not. Be clear and concise.

Analysis History:
{formatted_history}

Please provide:
1. A clear conclusion (SAME PERSON or DIFFERENT PERSONS)
2. Key evidence supporting your decision
3. Confidence level (High/Medium/Low)
"""
        final_summary = llm.text_to_text("", final_summary_prompt)[0]
        print("\n📌 Final Conclusion:")
        print(final_summary)
        break

    question = next_question
    input("Press Enter to continue to the next round...")

print("\n✅ Analysis completed!")
print(f"Total rounds: {len(history)}")
print(f"Questions asked: {[q for q, _, _ in history]}")