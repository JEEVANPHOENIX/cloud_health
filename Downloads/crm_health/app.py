import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ======================
# LOAD PUBLIC MODEL (WORKS IN CLOUD)
# ======================
model_name = "bhadresh-savani/distilbert-base-uncased-emotion"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# emotion labels (from model)
id2label = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# ======================
# SAFETY CHECK (VERY IMPORTANT)
# ======================
def safety_check(text):
    crisis_keywords = [
        "suicide", "kill myself", "jump", "die",
        "end my life", "want to die"
    ]
    return any(word in text.lower() for word in crisis_keywords)

# ======================
# PREDICT
# ======================
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    pred = torch.argmax(outputs.logits, dim=1).item()
    
    return id2label.get(pred, "neutral")

# ======================
# RESPONSE GENERATOR
# ======================
def generate_response(text):
    
    # 🚨 SAFETY FIRST
    if safety_check(text):
        return "⚠️ I'm really concerned about you. Please talk to someone immediately or contact a helpline 💙"

    emotion = predict(text)

    # responses based on emotion
    if emotion == "joy":
        return "😊 That's great to hear! Keep enjoying your day."
    
    elif emotion == "sadness":
        return "I'm really sorry you're feeling this way. You're not alone 💙"
    
    elif emotion == "anger":
        return "It seems frustrating. Try taking a deep breath and pause for a moment."
    
    elif emotion == "fear":
        return "It's okay to feel scared sometimes. You're stronger than you think."
    
    else:
        return "I understand how you're feeling. I'm here for you."

# ======================
# UI (CHAT STYLE)
# ======================
st.set_page_config(page_title="Mental Health AI", page_icon="🧠")

st.markdown(
    "<h1 style='text-align:center;color:#4CAF50;'>🧠 Mental Health Assistant</h1>",
    unsafe_allow_html=True
)

# chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# input
user_input = st.chat_input("How are you feeling today?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    response = generate_response(user_input)

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
