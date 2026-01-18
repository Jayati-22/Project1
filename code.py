# ============================================================
# WhatsApp → Stranger Things Character Mapper (Prototype)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 1. Main processing function
# -------------------------------
def process_whatsapp_file(file_path):
    """
    Takes a WhatsApp chat text file and a train.txt file,
    returns a DataFrame mapping WhatsApp users to their best-matched Stranger Things character.
    """
    
    # -------------------------------
    # Helper: Load WhatsApp chat into DataFrame grouped by user
    # -------------------------------
    def txt_to_df_test_grouped(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            text = " ".join(f.readlines())

        # WhatsApp message regex (date, time - sender: message)
        messages = re.split(
            r"\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s?[ap]m\s-\s",
            text
        )

        rows = []
        current_sender = None
        current_text = []

        for msg in messages[1:]:
            if ":" not in msg:
                continue

            sender, message = msg.split(":", 1)
            sender = sender.strip().upper()
            message = message.strip()

            # Skip empty or omitted messages
            if not message or "omitted" in message.lower():
                continue

            if sender == current_sender:
                current_text.append(message)
            else:
                if current_sender:
                    rows.append({
                        "WhatsApp_User": current_sender,
                        "Dialogue": " ".join(current_text)
                    })
                current_sender = sender
                current_text = [message]

        # Add last sender
        if current_sender and current_text:
            rows.append({
                "WhatsApp_User": current_sender,
                "Dialogue": " ".join(current_text)
            })

        return pd.DataFrame(rows)

    # -------------------------------
    # Helper: Load training dialogue
    # -------------------------------
    def txt_to_df_train(txt_path):
        data = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    character, dialogue = line.split(":", 1)
                    data.append({
                        "Character": character.strip().upper(),
                        "Dialogue": dialogue.strip()
                    })
        return pd.DataFrame(data)

    # -------------------------------
    # Load data
    # -------------------------------
    train_txt_path = './train.txt'
    df_train = txt_to_df_train(train_txt_path)
    df_test  = txt_to_df_test_grouped(file_path)

    # -------------------------------
    # Sentence embeddings
    # -------------------------------
    embedder = SentenceTransformer("all-mpnet-base-v2")

    train_embeddings = embedder.encode(df_train["Dialogue"].tolist())
    test_embeddings  = embedder.encode(df_test["Dialogue"].tolist())

    # -------------------------------
    # Average embedding per character
    # -------------------------------
    character_embeddings = {}
    for character in df_train["Character"].unique():
        mask = df_train["Character"] == character
        character_embeddings[character] = train_embeddings[mask].mean(axis=0)

    # -------------------------------
    # Similarity matching
    # -------------------------------
    rows = []
    for i, row in df_test.iterrows():
        user = row["WhatsApp_User"]
        emb = test_embeddings[i].reshape(1, -1)

        for character, char_emb in character_embeddings.items():
            sim = cosine_similarity(emb, char_emb.reshape(1, -1))[0][0]

            rows.append({
                "WhatsApp_User": user,
                "Character": character,
                "Similarity": sim
            })

    df_scores = pd.DataFrame(rows)

    # -------------------------------
    # Select best match per user
    # -------------------------------
    final_results = (
        df_scores
        .sort_values("Similarity", ascending=False)
        .groupby("WhatsApp_User", as_index=False)
        .first()
        .rename(columns={"Character": "Matched_Character"})
    )

    # final_results["Similarity"] = final_results["Similarity"].round(3)

    return final_results

# -------------------------------
# 2. Streamlit app
# -------------------------------
st.title("WhatsApp → Stranger Things Character Mapper")

st.markdown("""
Upload a WhatsApp exported chat text file. 
The app will match each user to their closest Stranger Things character.
""")

# File uploader for WhatsApp chat
uploaded_file = st.file_uploader("Upload WhatsApp chat (.txt)", type="txt")

# Optional: allow user to provide training file path
train_file_path = st.text_input(
    "Training file path (train.txt with Stranger Things dialogues)",
    value="train.txt"
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp_chat.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the chat
    with st.spinner("Matching characters... This may take a while for long chats."):
        df_results = process_whatsapp_file("temp_chat.txt", train_file_path)

    st.subheader("Mapped Characters")
    st.dataframe(df_results)
