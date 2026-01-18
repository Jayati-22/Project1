# # ============================================================
# # WhatsApp → Stranger Things Character Mapper (Prototype)
# # ============================================================

# import streamlit as st
# import pandas as pd
# import numpy as np
# import re
# from fastembed import TextEmbedding
# from sklearn.metrics.pairwise import cosine_similarity

# # -------------------------------
# # 1. Main processing function
# # -------------------------------
# def process_whatsapp_file(file_path):
#     """
#     Takes a WhatsApp chat text file and a train.txt file,
#     returns a DataFrame mapping WhatsApp users to their best-matched Stranger Things character.
#     """
    
#     # -------------------------------
#     # Helper: Load WhatsApp chat into DataFrame grouped by user
#     # -------------------------------
#     def txt_to_df_test_grouped(txt_path):
#         with open(txt_path, "r", encoding="utf-8") as f:
#             text = " ".join(f.readlines())

#         # WhatsApp message regex (date, time - sender: message)
#         messages = re.split(
#             r"\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s?[ap]m\s-\s",
#             text
#         )

#         rows = []
#         current_sender = None
#         current_text = []

#         for msg in messages[1:]:
#             if ":" not in msg:
#                 continue

#             sender, message = msg.split(":", 1)
#             sender = sender.strip().upper()
#             message = message.strip()

#             # Skip empty or omitted messages
#             if not message or "omitted" in message.lower():
#                 continue

#             if sender == current_sender:
#                 current_text.append(message)
#             else:
#                 if current_sender:
#                     rows.append({
#                         "WhatsApp_User": current_sender,
#                         "Dialogue": " ".join(current_text)
#                     })
#                 current_sender = sender
#                 current_text = [message]

#         # Add last sender
#         if current_sender and current_text:
#             rows.append({
#                 "WhatsApp_User": current_sender,
#                 "Dialogue": " ".join(current_text)
#             })

#         return pd.DataFrame(rows)

#     # -------------------------------
#     # Helper: Load training dialogue
#     # -------------------------------
#     def txt_to_df_train(txt_path):
#         data = []
#         with open(txt_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 line = line.strip()
#                 if ":" in line:
#                     character, dialogue = line.split(":", 1)
#                     data.append({
#                         "Character": character.strip().upper(),
#                         "Dialogue": dialogue.strip()
#                     })
#         return pd.DataFrame(data)

#     # -------------------------------
#     # Load data
#     # -------------------------------
#     train_txt_path = './train.txt'
#     df_train = txt_to_df_train(train_txt_path)
#     df_test  = txt_to_df_test_grouped(file_path)

#     # -------------------------------
#     # Sentence embeddings
#     # -------------------------------
#     embedder = TextEmbedding()

#     train_embeddings = np.array(list(embedder.embed(df_train["Dialogue"].tolist())))
#     test_embeddings  = np.array(list(embedder.embed(df_test["Dialogue"].tolist())))

#     # -------------------------------
#     # Average embedding per character
#     # -------------------------------
#     character_embeddings = {}
#     for character in df_train["Character"].unique():
#         mask = df_train["Character"] == character
#         character_embeddings[character] = train_embeddings[mask].mean(axis=0)

#     # -------------------------------
#     # Similarity matching
#     # -------------------------------
#     rows = []
#     for i, row in df_test.iterrows():
#         user = row["WhatsApp_User"]
#         emb = test_embeddings[i].reshape(1, -1)

#         for character, char_emb in character_embeddings.items():
#             sim = cosine_similarity(emb, char_emb.reshape(1, -1))[0][0]

#             rows.append({
#                 "WhatsApp_User": user,
#                 "Character": character,
#                 "Similarity": sim
#             })

#     df_scores = pd.DataFrame(rows)

#     # -------------------------------
#     # Select best match per user
#     # -------------------------------
#     final_results = (
#         df_scores
#         .sort_values("Similarity", ascending=False)
#         .groupby("WhatsApp_User", as_index=False)
#         .first()
#         .rename(columns={"Character": "Matched_Character"})
#     )

#     # final_results["Similarity"] = final_results["Similarity"].round(3)

#     return final_results

# # -------------------------------
# # 2. Streamlit app
# # -------------------------------
# # -------------------------------
# # 2. Streamlit app
# # -------------------------------

# # Stranger Things Theme CSS
# st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@700&display=swap');

#     /* Main background */
#     .stApp {
#         background-color: #0e0e0e;
#         color: #dcdcdc;
#     }

#     /* Titles and Headers */
#     h1, h2, h3, h4, h5, h6 {
#         font-family: 'Libre Baskerville', serif;
#         color: #E71D36 !important;
#         text-shadow: 0 0 5px #E71D36, 0 0 10px #8B0000;
#         letter-spacing: 1px;
#     }
    
#     /* Specific Title Styling */
#     h1 {
#         font-size: 3.5rem !important;
#         text-transform: uppercase;
#         border-bottom: 2px solid #E71D36;
#         padding-bottom: 10px;
#     }

#     /* Text */
#     p, div, label {
#         color: #dcdcdc !important;
#         font-family: 'Courier New', monospace;
#     }
    
#     /* Buttons */
#     .stButton > button {
#         background-color: transparent !important;
#         color: #E71D36 !important;
#         border: 2px solid #E71D36 !important;
#         border-radius: 5px;
#         font-family: 'Libre Baskerville', serif;
#         text-transform: uppercase;
#         transition: all 0.3s ease;
#     }
#     .stButton > button:hover {
#         box-shadow: 0 0 10px #E71D36;
#         color: #fff !important;
#     }

#     /* File Uploader */
#     .stFileUploader {
#         border: 1px dashed #E71D36;
#         padding: 20px;
#         border-radius: 10px;
#     }

#     /* DataFrame/Table */
#     [data-testid="stDataFrame"] {
#         border: 1px solid #444;
#     }
    
#     /* Scrollbars */
#     ::-webkit-scrollbar {
#         width: 10px;
#         background: #0e0e0e;
#     }
#     ::-webkit-scrollbar-thumb {
#         background: #E71D36; 
#         border-radius: 5px;
#     }
#     </style>
# """, unsafe_allow_html=True)

# st.title("WhatsApp → Stranger Things Character Mapper")

# st.markdown("""
# Upload a WhatsApp exported chat text file. 
# The app will match each user to their closest Stranger Things character.
# """)

# # File uploader for WhatsApp chat
# uploaded_file = st.file_uploader("Upload WhatsApp chat (.txt)", type="txt")

# # Optional: allow user to provide training file path
# train_file_path = st.text_input(
#     "Training file path (train.txt with Stranger Things dialogues)",
#     value="train.txt"
# )

# if uploaded_file is not None:
#     # Save uploaded file temporarily
#     with open("temp_chat.txt", "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     # Process the chat
#     with st.spinner("Matching characters... This may take a while for long chats."):
#         df_results = process_whatsapp_file("temp_chat.txt")

#     st.subheader("Mapped Characters")
#     st.dataframe(df_results)

# ============================================================
# WhatsApp → Stranger Things Character Mapper (Prototype)
# ============================================================

# ============================================================
# WhatsApp → Stranger Things Character Mapper (Prototype)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import re
from fastembed import TextEmbedding
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
    embedder = TextEmbedding()

    train_embeddings = np.array(list(embedder.embed(df_train["Dialogue"].tolist())))
    test_embeddings  = np.array(list(embedder.embed(df_test["Dialogue"].tolist())))

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


import base64

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@700&display=swap');

    /* Main background */
    .stApp {
        background-image: url("data:image/jpg;base64,%s");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: #e0e0e0;
    }
    
    /* Overlay to improve readability */
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%%;
        height: 100%%;
        background-color: rgba(0, 0, 0, 0.6); 
        z-index: -1;
    }

    /* Titles and Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Libre Baskerville', serif;
        color: #ff3333 !important; /* Brighter red for contrast against dark blue */
        text-shadow: 0 0 10px #ff0000, 0 0 20px #8B0000;
        letter-spacing: 1px;
    }
    
    /* Specific Title Styling */
    h1 {
        font-size: 3.5rem !important;
        text-transform: uppercase;
        border-bottom: 2px solid #ff3333;
        padding-bottom: 10px;
    }

    /* Text */
    p, div, label, li, span {
        color: #e0e0e0 !important;
        font-family: 'Courier New', monospace;
        font-weight: bold;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: rgba(0, 0, 0, 0.5) !important;
        color: #ff3333 !important;
        border: 2px solid #ff3333 !important;
        border-radius: 5px;
        font-family: 'Libre Baskerville', serif;
        text-transform: uppercase;
        transition: all 0.3s ease;
        backdrop-filter: blur(2px);
    }
    .stButton > button:hover {
        box-shadow: 0 0 15px #ff0000;
        background-color: rgba(255, 51, 51, 0.1) !important;
        color: #fff !important;
    }

    /* File Uploader */
    .stFileUploader {
        border: 2px dashed #ff3333;
        padding: 20px;
        border-radius: 10px;
        background-color: rgba(0, 0, 0, 0.4);
    }

    /* DataFrame/Table */
    [data-testid="stDataFrame"] {
        border: 1px solid #ff3333;
        background-color: rgba(0, 0, 0, 0.6);
    }
    
    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 10px;
        background: #000;
    }
    ::-webkit-scrollbar-thumb {
        background: #ff3333; 
        border-radius: 5px;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)

try:
    set_png_as_page_bg('background.jpg')
except Exception as e:
    st.warning(f"Could not load background image: {e}")


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
        df_results = process_whatsapp_file("temp_chat.txt")

    st.subheader("Mapped Characters")
    st.dataframe(df_results)



