# SmartBible

This project allows users to search relevant passages in the Bible based on input. Semantic search techniques make the search more precise, and I plan to improve text generation to support "smart chatting" with the documents. In this first draft, hugging face models are used for creating the embeddings, and ChromaDB stores them locally.

---
## Sample PDF and text

I provided a sample pdf, but the full text can be found at: [The Book of Genesis](https://www.vatican.va/archive/bible/genesis/documents/bible_genesis_en.html#:~:text=The%20Book%20of%20Genesis&text=%5B1%3A1%5D%20In%20the,%22%3B%20and%20there%20was%20light.)

## Getting Started
<sup>Make sure you have Python installed</sup>
### 1. Clone the Repository:
```bash
git clone https://github.com/laureanolorenzo/Smart-Bible.git
```
### 2. Install dependencies (I'm using python 3.11)
```bash
pip install -r requirements.txt
```
### 3. You can now use the main feature, which is document retrieval. 

The 2 main models will download the first time, and a chromaDB instance will be created. If you have an EdenAI API key, you can create an "edenAIkey.txt" file, place your key in it, and then set the "chunks_only" variable to False in the "mainv1.py" file. This will allow you to get an interpretation of the retrieved passages from an LLM.

To start the process, run:
```bash
python mainv1.py 
```

---

####  Version 2

Main change: migration of vector database from local (chromaDB) to remote (Pinecone). 

#### Currently working on:
 
 A small application to run the chatbot from a server. Maybe I'll deploy it later.
