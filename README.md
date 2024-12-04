# LLama3 RAG Application

This project is a **Retrieval Augmented Generation (RAG)** application built using the **LLama3** language model, **Groq**, **HuggingFace Embeddings**, and **Streamlit** for the user interface. The application allows users to query documents using a powerful retrieval-based approach, generating accurate and contextually relevant answers from a document store.

## Features

- **Retrieval Augmented Generation (RAG)**: Combines document retrieval with language model generation to provide more accurate answers based on a provided context.
- **Integration with Groq**: Leverages the **Groq** API for efficient query processing using the LLama3 model.
- **HuggingFace Embeddings**: Utilizes **HuggingFace** embeddings for converting text documents into vector representations.
- **LangChain Workflow**: Utilizes **LangChain** for document splitting, embedding, and retrieval workflow.
- **Streamlit Interface**: Provides an interactive UI where users can input queries and retrieve answers based on document contents.

## Tech Stack

- **LLama3** (Groq): Used for powerful language model-based query processing.
- **HuggingFace Embeddings**: For transforming text documents into vector embeddings.
- **LangChain**: For chaining document loading, retrieval, and generation processes.
- **Chroma**: A vector database for storing document embeddings and enabling fast retrieval.
- **Streamlit**: A Python framework for building interactive web applications, used here to build the user interface.
- **Pandas**: For loading and processing documents from CSV.

## Requirements

### Dependencies
The project uses the following dependencies:

- **Python 3.9** (or higher)
- **Streamlit** for building the web UI
- **LangChain** for building the document processing and retrieval chain
- **HuggingFace Transformers** for sentence embeddings
- **Groq** for large-scale language model inference

To install the required dependencies, run:

```bash
pip install -r requirements.txt
