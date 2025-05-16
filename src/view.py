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

    questions = [
        "Do the eyes of the two individuals have similar size and shape?",
        "Is there a noticeable difference in the nose length and width between the two individuals?",
        "Are the mouths of the two individuals similar in terms of lip thickness and symmetry?",
        "Do the facial structures, such as the jaw and chin, appear similar?",
        "Do the individuals have similar eyebrow shapes, density, or gaps between brows?"
    ]

    qa_list = []
    for i, question in enumerate(questions):
        question_dir = os.path.join(index_dir, f"question_{i}")
        responses = []

        if os.path.exists(question_dir):
            for response_file in sorted(os.listdir(question_dir)):
                if response_file.startswith("response_"):
                    response_path = os.path.join(question_dir, response_file)
                    with open(response_path, "r") as f:
                        response_text = f.read().strip()
                    responses.append(f"{response_file}:\n{response_text}")
        else:
            responses.append("No responses found.")

        qa_list.append((question, "\n\n".join(responses)))

    return img1, img2, str(label), decide_text, qa_list


def gradio_ui():
    with gr.Blocks(title="Visualize Comparison QA") as demo:
        gr.Markdown("# 👥 Visual QA Comparison Viewer")

        with gr.Row():
            index = gr.Number(label="Sample Index", value=0)
            input_dir = gr.Textbox(label="Input Directory", placeholder="Path to input dir")
            dataset_name = gr.Textbox(label="Dataset Name", placeholder="Dataset name used in get_dataset")
            run_button = gr.Button("Load Example")

        with gr.Row():
            img1_output = gr.Image(label="Image 1")
            img2_output = gr.Image(label="Image 2")

        label_output = gr.Textbox(label="Label")
        decide_output = gr.Textbox(label="Decision")

        qa_blocks = [gr.Markdown() for _ in range(5)]

        def update_all(index, input_dir, dataset_name):
            img1, img2, label, decide, qa_list = load_example(int(index), input_dir, dataset_name)
            qa_texts = [f"### Q{i+1}: {q}\n\n```\n{a}\n```" for i, (q, a) in enumerate(qa_list)]
            return [img1, img2, label, decide] + qa_texts

        run_button.click(
            fn=update_all,
            inputs=[index, input_dir, dataset_name],
            outputs=[img1_output, img2_output, label_output, decide_output] + qa_blocks
        )

    demo.launch()


if __name__ == "__main__":
    gradio_ui()
