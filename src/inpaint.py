import gradio as gr
from PIL import Image
from datetime import datetime


def save_drawn_image(drawing_dict):
    image = drawing_dict['composite']  # đây là ảnh đã vẽ xong
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"drawn_{timestamp}.png"
    image.convert("RGB").save(save_path)
    return image


with gr.Blocks() as demo:
    gr.Markdown("## ✏️ Vẽ màu trắng lên ảnh (sẽ lưu lại ảnh sau khi vẽ)")

    editor = gr.ImageEditor(
        type="pil",
        label="Vẽ bằng màu trắng (chọn brush trắng)",
    )

    output = gr.Image(type="pil", label="Ảnh đã vẽ (đã lưu)")

    btn = gr.Button("💾 Lưu ảnh đã vẽ")

    btn.click(save_drawn_image, inputs=editor, outputs=output)

demo.launch()
