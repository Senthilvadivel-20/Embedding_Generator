import streamlit as st
import pandas as pd


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

#Our sentences we like to encode
# sentences = ['This framework generates embeddings for each input sentence',
#     'Sentences are passed as a list of string.', 
#     'The quick brown fox jumps over the lazy dog.']

#Sentences are encoded by calling model.encode()
def get_embeddings(sentence):
    embeddings = model.encode(sentence)
    return embeddings

#Print the embeddings
# for sentence, embedding in zip(sentences, embeddings):
#     print("Sentence:", sentence)
#     print("Embedding:", embedding)
#     print("")

st.set_page_config(
    page_title="Embeddings generator ", page_icon=":bird:")

st.header("Type anything :bird:")
message = st.text_area("Type something")

if message:
    st.write("Embeddings vectors for your input.....")

    result = get_embeddings(message)

    st.info(result)