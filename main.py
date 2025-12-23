import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_MODEL = "snithshibu/mistral-em-sadiku-lora"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)

def predict(message, history):
    formatted_prompt = f"[INSTRUCTION]\nExplain the following electromagnetics content clearly.\n\n[INPUT]\n{message}\n\n[RESPONSE]\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("[RESPONSE]\n")[-1]
    return response

demo = gr.ChatInterface(
    fn=predict,
    title="⚡ Electromagnetics Expert AI",
    description="Ask me anything about Electromagnetics! Fine-tuned on Sadiku's textbook.",
)

if __name__ == "__main__":
    demo.launch()
