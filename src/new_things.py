# from text_to_text_services import LlamaService
# from get_architech import init_lvlm_model
# from PIL import Image

# llm = LlamaService(model_name="Llama-7b")
# lvlm_model, lvlm_image_token, lvlm_special_token = init_lvlm_model("llava-onevision-qwen2-7b-ov", "llava_qwen")
# img_files = [Image.open("D:\Enhanced-Evolutionary-Black-box-Patch-Attacks-on-Face-Verification\sontung_2.png"),
#              Image.open("D:\Enhanced-Evolutionary-Black-box-Patch-Attacks-on-Face-Verification\sontung_1.png")]

# history = []

# init_lvlm_prompt = "Do these two facial images belong to the same person? Please explain your reasoning."
# llm_system_prompt = """
# You are analyzing whether two facial images belong to the same person. 
# You have received the following Q&A history.
# Based on the previous answers and questions, please propose a follow-up question that explores this reasoning in more detail.
# If enough information has been gathered, please respond with "None".
# """
# llm_prompt = "History: {history}"


# q = init_lvlm_model
# round = 0
# T = 10
# while round < T:
#     a = lvlm_model.inference(
#         q + lvlm_image_token * 2,
#         img_files, num_return_sequences=1,
#         do_sample=True, temperature=0.8, reload=False
#     )[0]
    
#     history.append((q, a))
#     q_next = llm.text_to_text(llm_system_prompt, llm_prompt.format(history=history))
    
#     if "None" in q_next:
#         break
    
#     q = q_next
#     round += 1
#     print("Round: ", round, 
#           "Question: ", q, 
#           "Answer: ", a)
    
#     input()
    

from text_to_text_services import LlamaService
from get_architech import init_lvlm_model
from PIL import Image

llm = LlamaService(model_name="Llama-7b")
lvlm_model, lvlm_image_token, lvlm_special_token = init_lvlm_model("llava-onevision-qwen2-7b-ov", "llava_qwen")

img_files = [
    Image.open("../sontung_2.png"),
    Image.open("../sontung_1.png")
]

initial_question = "Do these two facial images belong to the same person? Please explain your reasoning."

llm_system_prompt = """
You are analyzing whether two facial images belong to the same person.
You have received the following Q&A history.
Based on the previous answers and questions, please propose a follow-up question that explores this reasoning in more detail.
If enough information has been gathered, please respond with "None".
"""

llm_prompt_template = "History:\n{history}"

history = []
question = initial_question
max_rounds = 10

for round_idx in range(max_rounds):
    answer = lvlm_model.inference(
        question + lvlm_image_token * 2,
        img_files,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8,
        reload=False
    )[0]

    history.append((question, answer))

    formatted_history = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])

    next_question = llm.text_to_text(llm_system_prompt, llm_prompt_template.format(history=formatted_history))

    print(f"\n🔁 Round {round_idx + 1}")
    print(f"❓ Question: {question}")
    print(f"✅ Answer: {answer}")

    if "None" in next_question:
        final_summary_prompt = f"""
Based on the following Q&A history, please summarize the reasoning and conclude whether the two faces belong to the same person or not. Be clear and concise.

{formatted_history}
"""
        final_summary = llm.text_to_text("", final_summary_prompt)
        print("\n📌 Final Conclusion:\n", final_summary)
        break

    question = next_question
    input("Press Enter to continue to the next round...")


    

    
    
    
    
    

