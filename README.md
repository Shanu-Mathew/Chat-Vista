# Chat-Vista
This is an innovative conversational chatbot powered by the Llama-2 Large Language Model (LLM) and the Retrieval-Augmented Generation (RAG) architecture. This advanced chatbot is designed to provide intelligent and contextually relevant responses by leveraging state-of-the-art natural language processing techniques based on the documents uploaded.

## Overview
Chat-Vista allows users to upload documents in various formats such as PDF, TXT, and DOCX. The uploaded documents are processed and used to create a conversational model. Users can then interact with the model by asking questions or initiating conversations.

## Technology Used
- **Streamlit**: A Python library for creating web applications with minimal effort.

- **LangChain**: A library for building conversational AI systems with various components like embeddings, text splitters, vector stores, and more.

- **Hugging Face Embeddings**: Utilized for creating embeddings using the Sentence Transformers library.

- **FAISS**: A library for efficient similarity search and clustering of dense vectors. It stands for Facebook AI Similarity Search.

## Features
- **Document Upload**: Users can upload PDF, TXT, and DOCX files containing information relevant to the conversation.

- **Conversational Model**: The application uses LangChain to create a Conversational Retrieval Chain, which is capable of generating responses based on the uploaded documents.

- **Interactive Chat Interface**: Users can ask questions and engage in conversations with the model through a user-friendly interface.

## How to Run
To run the Chat-Vista application, follow these steps:

Prerequisites
Make sure you have Python installed on your machine.

#### 1) Clone the Repository:

```python
git clone https://github.com/Shanu-Mathew/Chat-Vista.git
```
#### 2) Install Dependencies:
```python
pip install -r requirements.txt
```
#### 3) Download Model:
The models folder containing the language models is not included in this repository due to its size. Create a models folder and download the model into it. The model used is llama-2 7 billion parameter ggml model by The Bloke taken from HuggingFace. The model can be accessed [here](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML)

#### 4) Run the Application:
```python
streamlit run chat_vista.py
```
This command will start the Streamlit application, and you can access it in your web browser at http://localhost:8501.

#### 5) Usage:
Open the application in your web browser. Use the sidebar to upload documents (PDF, TXT, DOCX). The application will process the documents and create a vector database. Interact with the model by typing questions in the input field. The model will provide intelligent responses based on the uploaded documents.

## Acknowledgments
Special thanks to LangChain and Hugging Face for their incredible tools and libraries.
