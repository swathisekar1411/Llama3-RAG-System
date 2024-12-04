# nlp_utils.py
import os
import numpy as np
import random
import torch
from tqdm import tqdm
import pandas as pd
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from uuid import uuid4
from langchain.docstore.document import Document


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_embeddings_and_store(
    embeddings_model="BAAI/bge-small-en-v1.5", csv_path="./documents.csv"
):
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    vector_store = Chroma(
        collection_name="my_collection",
        embedding_function=hf_embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n\n",
        chunk_size=1000,
    )
    if vector_store._collection.count() == 0:
        print("Adding documents to the database...")

        df = pd.read_csv(csv_path, index_col=0)

        docs = []
        for i, row in tqdm(df.iterrows()):
            doc_chunks = text_splitter.split_text(row["text"])
            docs.extend(
                [
                    Document(
                        page_content=chunk,
                        metadata={"index": i, "source_url": row["source_url"]},
                    )
                    for chunk in doc_chunks
                ]
            )

        uuids = [str(uuid4()) for _ in range(len(docs))]
        vector_store.add_documents(documents=docs, ids=uuids)
    else:
        print("Database already exists. Skipping document addition.")
        print(f"total documents: {vector_store._collection.count()}")
    return hf_embeddings, vector_store


def setup_retrieval_chain(vector_store, groq_api_key, llm_name="llama-3.2-1b-preview"):
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=llm_name,
    )
    prompt = ChatPromptTemplate.from_template(
        """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Questions:{input}
    """
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain


def search_document(retrieval_chain, question):
    results = retrieval_chain.invoke({"input": question})
    return results
