import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ======================
# LOAD MODEL
# ======================
model_path = "mental_health_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# ======================
# LABEL MAP (EDIT IF NEEDED)
# ======================
id2label = {
    0: "anxiety_low",
    1: "anxiety_medium",
    2: "anxiety_high",
    3: "sadness_low",
    4: "sadness_medium",
    5: "sadness_high",
    6: "joy_low",
    7: "neutral_low"
}

# ======================
# SAFETY CHECK
# ======================
def safety_check(text):
    crisis_keywords = [
        "suicide", "kill myself", "jump", "die",
        "end my life", "want to die"
    ]
    return any(word in text.lower() for word in crisis_keywords)

# ======================
# RULE BOOST
# ======================
def rule_boost(text):
    text = text.lower()
    if "happy" in text or "excited" in text:
        return "joy_low"
    if "sad" in text or "empty" in text:
        return "sadness_medium"
    return None

# ======================
# PREDICT
# ======================
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return id2label.get(pred, "neutral_low")

# ======================
# RESPONSE
# ======================
def generate_response(text):
    if safety_check(text):
        return "⚠️ Please reach out to someone immediately. You are not alone."

    label = rule_boost(text) or predict(text)
    emotion, severity = label.split("_")

    if severity == "high":
        return "I'm really sorry you're feeling this way. Please talk to someone you trust."
    elif severity == "medium":
        return "It seems challenging. Try breathing exercises and take breaks."
    else:
        return "You're doing okay. Keep taking care of yourself."

# ======================
# UI DESIGN (CHATGPT STYLE)
# ======================
st.set_page_config(page_title="Mental Health AI", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🧠 Mental Health Assistant</h1>
    <p style='text-align: center;'>Talk freely. I'm here to help 💙</p>
""", unsafe_allow_html=True)

# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# input
user_input = st.chat_input("How are you feeling today?")

if user_input:
    # user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # bot response
    response = generate_response(user_input)

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
