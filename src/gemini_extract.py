import argparse
from qwen.model import QWENModel
from tqdm import tqdm


def main(args):
    
    llm = QWENModel()
    system_prompt = f"""You are an AI assistant specialized in text analysis. Your task is to read the provided text, which is the output of another AI analyzing whether two images depict the same person.
    Based on the text content, determine if the final conclusion is 'same person' or 'different person'.
    Respond *only* with one of the following numbers:
    - Respond with 0 if the conclusion is SAME PERSON.
    - Respond with 1 if the conclusion is DIFFERENT PERSON.
    DO NOT add any explanation, greeting, or any other characters besides the number 0 or 1.
    """
    
    
    with open(args.response_path, "r") as f:
        responses = [line.strip() for line in f.readlines()]
    
    output_path = args.response_path.replace(".txt", "_processed.txt")
    
    
    
    with open(output_path, "w") as f:
        for i in tqdm(range(0, len(responses), args.batch_size)):
            batch_response = responses[i:i + args.batch_size]

            output_llm = llm.inference(prompts=batch_response, 
                                       system_prompt=system_prompt)

            for out in output_llm:
                f.write(f"{out}\n")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default="8")
    args = parser.parse_args()
    main(args)