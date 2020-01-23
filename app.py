import streamlit as st
import transformers as tr
import numpy as np
import torch
import pandas as pd

emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']


def call_model(text: str, tokenizer: tr.PreTrainedTokenizer, model: tr.PreTrainedModel):
    text = tokenizer.encode(text)
    x = np.full((1, 128), fill_value=tokenizer.pad_token_id, dtype=np.float64)
    attn = np.zeros((1, 128))
    for i, tok in enumerate(text):
        x[0][i] = tok
        attn[0][i] = 1

    with torch.no_grad():
        x = torch.from_numpy(x).to(torch.int64)
        attn = torch.from_numpy(attn)
        logits, = model(x, attention_mask=attn)
        probs = torch.sigmoid(logits).cpu().numpy()

    result = []
    for i, prob in enumerate(probs[0]):
        result.append({'emotion': emotions[i], 'intensity': prob})

    result = reversed(sorted(result, key=lambda x: x['intensity']))

    return pd.DataFrame(result)


model_dir: str = "./albert-model/"
tokenizer: tr.AlbertTokenizer = tr.AlbertTokenizer.from_pretrained(model_dir)
model: tr.AlbertForSequenceClassification = tr.AlbertForSequenceClassification.from_pretrained(model_dir)

st.title("ALBERT-base-v2 Emotions recognition")

st.header("Enter some text, and see what emotions are inside:")
text = st.text_input("Text to detect some emotions", "I like you")
st.write("There are emotions we found: ", call_model(text, tokenizer, model))