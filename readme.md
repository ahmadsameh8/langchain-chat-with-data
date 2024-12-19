# chat with data using langchain

the projet uses falcon3-1b with chromadb through langchain to chat with custom data



## Table of Contents

- [Description](#description)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Running the Project](#running-the-project)


## Description

This notebook demonstrates a workflow for creating a chatbot with custom data. It begins by using ```PyPDFLoader``` to load a PDF file from a specified URL and extract its content into individual pages. Next, it loads three CSV datasets and performs operations to clean and process the data, replacing missing values and preparing it for further use. The data is then transformed into document-like structures suitable for retrieval-augmented generation (RAG). The processed datasets and PDF pages are merged into a single collection, and the document data is embedded using the default ```HuggingFaceEmbeddings``` model. ```ChromaDB``` is created to store these embeddings for fast similarity searches. The workflow then utilizes the ```Falcon-3B-Instruct``` model from Hugging Face for text generation, implementing a conversational system with retrieval capabilities. ```ConversationBufferMemory``` from LangChain is incorporated to retain memory, enabling follow-up questions. Finally, the Conversational Retrieval Chain combines the language model, vector database retriever, and memory to enable context-aware question-answering.

After that a Flask Api is created combining all the previously mentioned steps to create an effective chatbot


## Getting Started

### Prerequisites

- Python 3.11.10
- [ChromaDB weights]([https://www.google.com/drive/](https://drive.google.com/file/d/1HmRp8nBQGOHuJSrdy1BApQKIdGjb5FpF/view?usp=sharing)) 


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ahmadsameh8/langchain-chat-with-data.git
   ```
2. Navigate to the project directory:
```bash
   cd langchain-chat-with-data
```
3. Install the required packages:
```bash
pip install -r req.txt
```

# Running the Project
After setting up the environment, run:
```bash
python app.py
```
This will start your main application with the flask api.

# To test your application, use the test_app.ipynb Jupyter Notebook. Open it and send requests as needed.


