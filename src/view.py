import os
import gradio as gr
from PIL import Image
from dataset import get_dataset


def load_example(index, input_dir, dataset_name):
    dataset = get_dataset(dataset_name)
    img1, img2, label = dataset[index]

    index_dir = os.path.join(input_dir, str(index))

    # Load decide.txt
    decide_path = os.path.join(index_dir, "decide.txt")
    decide_text = open(decide_path, "r").read().strip() if os.path.exists(decide_path) else "Not found"

    # Load question & response
    qa_list = []
    for i in range(5):  # assuming 5 questions
        question_path = os.path.join(index_dir, f"question_{i}.txt")
        if os.path.exists(question_path):
            with open(question_path, "r") as f:
                line = f.readline().strip()
                if "\t" in line:
                    question, response = line.split("\t", 1)
                    qa_list.append((question, response))
                else:
                    qa_list.append(("Invalid format", line))
        else:
            qa_list.append((f"question_{i}.txt not found", ""))

    return img1, img2, str(label), decide_text, qa_list


def gradio_ui():
    def show(index, input_dir, dataset_name):
        img1, img2, label, decide, qa_list = load_example(index, input_dir, dataset_name)
        questions = "\n".join([f"Q{i+1}: {q}\n→ {a}" for i, (q, a) in enumerate(qa_list)])
        return img1, img2, label, decide, questions

    gr.Interface(
        fn=show,
        inputs=[
            gr.Number(label="Sample Index", value=0),
            gr.Textbox(label="Input Directory", placeholder="Path to input dir"),
            gr.Textbox(label="Dataset Name", placeholder="Dataset name used in get_dataset"),
        ],
        outputs=[
            gr.Image(label="Image 1"),
            gr.Image(label="Image 2"),
            gr.Textbox(label="Label"),
            gr.Textbox(label="Decision"),
            gr.Textbox(label="Questions and Responses", lines=10),
        ],
        title="Visualize Comparison QA"
    ).launch()


if __name__ == "__main__":
    gradio_ui()
