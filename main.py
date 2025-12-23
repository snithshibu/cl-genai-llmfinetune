import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

st.set_page_config(page_title="EM Expert AI", page_icon="⚡")

st.title("⚡ Electromagnetics Expert AI")
st.markdown("Fine-tuned on Sadiku's Elements of Electromagnetics")

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_MODEL = "snithshibu/mistral-em-sadiku-lora"

@st.cache_resource
def load_model():
    st.info("Loading model... this may take a minute.")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
    return model, tokenizer

model, tokenizer = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask an Electromagnetics question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        formatted_prompt = f"[INSTRUCTION]\nExplain the following electromagnetics content clearly.\n\n[INPUT]\n{prompt}\n\n[RESPONSE]\n"
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("[RESPONSE]\n")[-1]
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
