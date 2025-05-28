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

initial_question = "Look at both images. Describe the key facial features you can observe in each image."

llm_system_prompt = """
🎮 DETECTIVE CHALLENGE: Determine if two faces show the same person using strategic questions!

🕵️ Your Mission: You're a master detective analyzing two face images. A Vision AI will examine BOTH images simultaneously and answer your questions about what it observes.

🎯 GAME RULES:
- Ask strategic questions about features visible in both images
- The Vision AI can see and compare both images directly
- Each question costs points - fewer questions = higher score!
- Analyze the responses to determine if it's the same person or different people
- When you're confident about your conclusion, respond with "CONCLUSION"
- Focus on distinguishing facial features, expressions, angles, lighting, etc.

⚠️ STRATEGY TIPS: 
- Ask about specific, measurable features (eye color, nose shape, facial structure, etc.)
- Consider lighting differences, angles, and image quality
- Look for unique identifying features (scars, moles, distinctive shapes)
- Don't just ask "are they the same person" - gather evidence first!

⚡ Only return your next strategic question. Nothing else. If you have enough evidence to make a final determination, return "CONCLUSION".

What's your next detective question?
"""

llm_prompt_template = "Investigation History:\n{history}\n\nWhat's your next strategic question to solve this case?"

history = []
question = initial_question
max_rounds = 10

for round_idx in range(max_rounds):
    # Ask the question about BOTH images simultaneously
    vision_response = lvlm_model.inference(
        question + f" {lvlm_image_token} {lvlm_image_token}",  # Two image tokens for both images
        img_files,  # Pass both images
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8,
        reload=False
    )[0]
    
    # Store the question and response in history
    history.append((question, vision_response))

    # Format history for LLM
    formatted_history = ""
    for i, (q, response) in enumerate(history, 1):
        formatted_history += f"Round {i}\nQ: {q}\nVision AI Response: {response}\n\n"

    # Generate next question based on the response
    next_question = llm.text_to_text(llm_system_prompt, llm_prompt_template.format(history=formatted_history))[0]

    print(f"\n🎮 Detective Round {round_idx + 1}")
    print(f"🕵️ Detective Question: {question}")
    print(f"👁️ Vision AI Analysis: {vision_response}")
    print(f"🤔 Detective's Next Move: {next_question}")

    if "CONCLUSION" in next_question.upper():
        print("\n🎯 REACHING FINAL CONCLUSION!")
        
        # Final analysis prompt
        final_analysis_prompt = f"""
Time for your final detective conclusion! You've been investigating whether two face images show the same person or different people.

Here's your complete investigation:
{formatted_history}

Based on all the evidence gathered, provide your final analysis:

🎯 FINAL VERDICT: Are these the SAME PERSON or DIFFERENT PEOPLE?

🔍 KEY EVIDENCE: What specific features and observations led to your conclusion?

📊 CONFIDENCE LEVEL: How confident are you in this determination? (High/Medium/Low)

🎮 DETECTIVE SUMMARY: What were the most crucial clues that solved this case?

⚖️ REASONING: Explain your logical deduction process step by step.
"""
        
        final_verdict = llm.text_to_text("", final_analysis_prompt)[0]
        print("\n🏆 DETECTIVE'S FINAL VERDICT:")
        print(final_verdict)
        break

    question = next_question
    input("Press Enter to continue to next round...")

print("\n🎮 INVESTIGATION COMPLETED!")
print(f"🔢 Total Rounds: {len(history)}")
print(f"📝 Questions Asked: {len(history)}")
for i, (q, _) in enumerate(history, 1):
    print(f"   {i}. {q}")