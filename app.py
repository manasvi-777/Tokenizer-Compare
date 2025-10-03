import streamlit as st
from transformers import AutoTokenizer
import tiktoken

@st.cache_resource
def load_tokenizers():
    return {
        "GPT-3.5 / GPT-4 (tiktoken)": tiktoken.get_encoding("cl100k_base"),
        "BERT (WordPiece)": AutoTokenizer.from_pretrained("bert-base-uncased"),
        "T5 (SentencePiece)": AutoTokenizer.from_pretrained("t5-base"),
        "Grok-1 (SentencePiece)": AutoTokenizer.from_pretrained("Xenova/grok-1-tokenizer"),
        # Optionally:
        # "BLOOM": AutoTokenizer.from_pretrained("bigscience/bloom"),
        # "GPT-2": AutoTokenizer.from_pretrained("gpt2"),
    }

tokenizers = load_tokenizers()

st.title("🔍 Tokenizer Comparison (Open Access Models Only)")
st.write("Compare how open / public models tokenize text.")

user_text = st.text_area("Enter text:", "The book is good.")

if st.button("Compare"):
    cols = st.columns(len(tokenizers))
    for (name, tok), col in zip(tokenizers.items(), cols):
        with col:
            st.subheader(name)
            if "GPT" in name:
                tokens = tok.encode(user_text)
                decoded = [tok.decode([t]) for t in tokens]
            else:
                tokens = tok.encode(user_text)
                decoded = tok.convert_ids_to_tokens(tokens)

            st.write(decoded)
            st.caption(f"Count: {len(tokens)}")
