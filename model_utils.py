# model_utils.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def load_text_model(model_name="gpt2"):
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    return generator

def generate_answer(generator, caption, question):
    prompt = f"Image: {caption}\nQuestion: {question}\nAnswer:"
    out = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
    text = out[0]["generated_text"]
    if "Answer:" in text:
        return text.split("Answer:")[-1].strip()
    return text
