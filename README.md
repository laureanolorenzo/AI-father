# SmartBible

This project allows users to search relevant passages in the Bible based on input. Semantic search techniques make the search more precise, and I plan to improve text generation to support "smart chatting" with the documents. In this first draft, hugging face models are used for creating the embeddings, and ChromaDB stores them locally. Please note that I am not responsible for the answers. They are automatically fetched by a semantic retrieval system.

---

### Getting Started

## Check out the live Demo at [Smart Bible](https://sm-bible-remote.onrender.com/)

--- 


Commands below are currently not working!


---

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

### Current Version

#### Main changes: 

+ App is now live at [Smart Bible](https://sm-bible-remote.onrender.com/)

+ Added a GUI that can be used locally (access to Pinecone Index is needed, so it won't work yet)

+ A few minor changes to the RAG system itself

+ Added a JSON file which contains the chunks of text (original source: [Bible pdfs](https://github.com/christislord12/Bible-Pdfs/tree/main/pdfs))

#### Currently working on:
 
 + Adapting local version via ChromaDB

 + Looking for cheaper hosting alternatives to keep the app deployed

