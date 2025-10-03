# app.py
import gradio as gr
from captioner import image_to_caption
from model_utils import load_text_model, generate_answer

generator = load_text_model("gpt2")

def chatbot(image, question):
    if image is None:
        return "No image", "Please upload an image"
    caption = image_to_caption(image)
    if not question.strip():
        return caption, "Please ask a question"
    answer = generate_answer(generator, caption, question)
    return caption, answer

with gr.Blocks() as demo:
    gr.Markdown("## 🖼️ Image + Text Chatbot")
    img = gr.Image(type="pil", label="Upload an Image")
    q = gr.Textbox(label="Ask about the image")
    caption_out = gr.Textbox(label="Caption")
    answer_out = gr.Textbox(label="Answer")
    btn = gr.Button("Ask")
    btn.click(chatbot, inputs=[img, q], outputs=[caption_out, answer_out])

if __name__ == "__main__":
    demo.launch()
