# captioner.py
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def image_to_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption
