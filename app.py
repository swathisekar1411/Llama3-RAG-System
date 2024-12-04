# app.py
import streamlit as st
from RAGEngine import (
    seed_everything,
    setup_embeddings_and_store,
    setup_retrieval_chain,
    search_document,
)

# Set the seed for reproducibility
seed_everything(0)

# Set up the embeddings and vector store
hf_embeddings, vector_store = setup_embeddings_and_store(
    embeddings_model="BAAI/bge-small-en-v1.5", csv_path="./documents.csv"
)
st.session_state.update({"hf_embeddings": hf_embeddings, "vector_store": vector_store})
# Set up the retrieval chain
retrieval_chain = setup_retrieval_chain(vector_store, groq_api_key=st.secrets.GROQ_API_KEY, llm_name="llama-3.2-1b-preview")

# Streamlit UI
st.title("Document Question Answering System")
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input("Ask a question:")

if question:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Retrieving answer..."):
        results = search_document(retrieval_chain, question)
        with st.chat_message("assistant"):
            st.write(results["answer"])
        st.session_state.messages.append({"role": "assistant", "content": results["answer"]})
