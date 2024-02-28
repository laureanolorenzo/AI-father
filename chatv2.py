#Imports 
from pdfminer.high_level import extract_text #Again, can improve a lot in terms of pdf processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import json
import requests
from pinecone_utils import *
import torch
torch.device('cuda:0')

with open('pineConeAPIkey.txt', 'r',encoding='utf-8') as f:
    pc_API_key = f.read()
f.close()


# Pinecone connection
doc_name = 'sb_test'
index_name='smart-bible'
new_index = pinecone_index(pc_API_key,index_name)




## Embeddings Models
from sentence_transformers import SentenceTransformer,CrossEncoder
transformer_model = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
cross_encoder_model = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
completions_model = 'gpt-3.5-turbo-0125' #Completions model

print('Loading models...')
embeddings_model = SentenceTransformer(transformer_model)
cross_encoder = CrossEncoder(cross_encoder_model)
print('Models loaded successfully!')


try: 
    stop_words = set(stopwords.words('english'))
except: #May need to install it if it's the first time you use it
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))




################################4

###Config Variables###
# IMPORTANT: their optimal values depend on the type of text that is being processed. 
# For instance: for news articles, smaller chunk sizes and passing more individual chunks to the API works better (for me at least).
# Text quality is also key. From the research, I've concluded that writing your own texts with simple and explicit information works best.

################################

chunks_only = True #Wether to create a text completion based on the relevant chunks

file_name = 'bible_sample' #Remember to change this variabke if you supply your own text
# Will serve as an identifier in the ChromaDB, (if the collection doesn't already contain embeddings with this filename 
# in their metadata, then the text extraction and embeddings creation process is started).
file_path = f'documents//{file_name}.pdf' 


chunk_size = 1000 #Experiment with this
chunk_overlap = 100 
window_len = 0 # "n". Controls the length of the conversation. If set to 0, a new conversation is created each time
# After some testing, I found a trade-off between being able to keep a history of the conversation, and 
# getting responses that are consistent with the context



n_results_passed = 5 #How many chunks are to be passed to the API. Be mindful when adjusting, since it affects token consumption
                    # Multiple smaller chunks worked best with the "tomato" pdf
n_results_extra = 1 #Not too important. Controls the number of "extra" chunks (lower scoress) that are shown after a response


### Threshold (important)
metric_threshold = 0.82 # If in any given interaction no chunk  reaches this degree of similarity to the question, then
# the API call is canceled and an automatic response is sent instead (serves as a filter). If you wish to always
# call the API, then set it to 0

###Another layer of protection against the chatbot responding unrelated questions
def filter_stopwords(stop_words, chunk,lang = 'english'):
    #Using nltk to remove stopwords. Supposedly improves embedding quality when searching
    filtered_words = []
    word_tokens = word_tokenize(chunk)
    for word in word_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    return ' '.join(filtered_words)





#####Token consumption#####




### Functions ###
### This section is meant to help keep the main loop short and simple, but I need to redesign the whole code to achieve that.
### Although it is an improvement from v1, I need to tidy up the code and clearly define classes (maybe a "chain" object). 
### Hopefully I can do so in future versions.


#Show context if answer isn't in document
def show_context(query_results,n_results_extra = 0,choice = 'yes',cross_scores = True):
    
    if choice in ['yes','yess','Yes','YES', 'y','Y']:
        print('*-----------------------')
        time.sleep(2)
        print('Relevant passages:')
        if cross_scores:
            for n, (text, cross_score,distance) in enumerate(zip(query_results['documents'][:n_results_passed],query_results['cross_scores'][:n_results_passed],query_results['distances'][:n_results_passed])): 
                text = ''.join([s for s in text.splitlines(True) if s.strip('\r\n')])
                print(f'{n+1}:\n{text}\nDistance: {1-distance}\nCross Score: {cross_score}')
                time.sleep(2)
        else: 
            for n, (text,distance) in enumerate(zip(query_results['documents'][:n_results_passed],query_results['distances'][:n_results_passed])): 
                text = ''.join([s for s in text.splitlines(True) if s.strip('\r\n')])
                print(f'{n+1}:\n{text}\nDistance: {1-distance}')
                time.sleep(2)
        if n_results_extra > 0 and cross_scores: #Useful to see resulting chunks when searching for optimal parameter configuration
            print('*-----------------------') #Fix later
            print('Might be of interest: ')
            time.sleep(5)
            for n, (text, cross_score) in enumerate(zip(query_results['documents'][n_results_passed:n_results_passed+n_results_extra],query_results['cross_scores'][n_results_passed:n_results_passed+n_results_extra])): 
                text = ''.join([s for s in text.splitlines(True) if s.strip('\r\n')])
                print(f'{n+1}:\n{text}\nCross Score: {cross_score}')

def semantic_search(query,index,re_rank = True, threshold = None,prompt = None,n=50,top_n = 10,lang='spanish',doc_name = 'sb_test'): #Should access a global variable "collection"
    filtered_query = filter_stopwords(stop_words,query,lang=lang)
    query_embeddings = embeddings_model.encode(filtered_query,prompt = prompt)
    query_embeddings = query_embeddings.tolist()
    if n > 50:
        n = 50 #50 is enough!
    query_results = index.query( #See https://docs.pinecone.io/reference/query
        namespace= doc_name,
        top_k=n,
        include_metadata=True,
        vector=[q for q in query_embeddings],
    # sparseVector={
    #     'indices': [10, 45, 16],
    #     'values':  [0.5, 0.5, 0.2]
    # },
    )
    
    query_results = query_results['matches']
    mapped_results = {
        'documents':list(map(lambda x:x['metadata']['text'],query_results)),
        'distances': list(map(lambda x:x['score'],query_results))
    }
    if not re_rank:
        return mapped_results #So far decent results. Ada model is still better though.
    second_result = cross_encoder.predict([[filtered_query,filtered_chunk] for filtered_chunk in mapped_results['documents']])
    mapped_results['cross_scores'] = []
    for idx in range(len(second_result)):
        mapped_results['cross_scores'].append(second_result[idx])
    sorted_items = sorted(zip(mapped_results['documents'],mapped_results['cross_scores'],mapped_results['distances']),key=lambda x:x[1],reverse=True) #Sort by second score (cross score)
    new_results = {'documents':[item[0] for item in sorted_items[:top_n]],'cross_scores':[item[1] for item in sorted_items[:top_n]],'distances':[item[2] for item in sorted_items[:top_n]]}
    return new_results




###Text splitting
character_splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    chunk_overlap = chunk_overlap, #No len function
    separators= ['[','\n['] #Optimized for the bible
)
###Eden AI4
with open('edenAIkey.txt','r') as f:
    eden_key = f.readlines()[0]
    eden_key = eden_key.replace('\n','')
f.close()
def get_completions_from_eden(relevant_chunks,query,n=1): 
    prompt = f'''You are a helpful AI assistant who interprets passages from the bible and helps the user uncover their deeper meaning. Passages: {'#'.join([p for p in relevant_chunks[:n]])}
    Your answer (limit yourself to interpreting the passages and think step by step): '''
    headers = {"Authorization": f"Bearer {eden_key}"}
    url = "https://api.edenai.run/v2/text/generation"
    payload = {
        "providers": "cohere",
        "text": prompt,
        "temperature": 0.2,
        "max_tokens": 150,
    }
    response = requests.post(url=url, json=payload, headers=headers)
    result = json.loads(response.text)
    return result['cohere']

### Not too important. This variables tries to limit the number of times the main loop runs with errors.
max_n_tries = 3 

## chromaDB client and collection setup



##Checking if the document is already in the collection

if not docs_exist(new_index):
    print(f'Extracting text from {doc_name}')
    try:
        text = extract_text(file_path) 
        # It handled the tomato file pretty well, but it couldn't deal with table in the earnings report.
    except:
        print('Error occurred while processing pdf. Please try again using a different file. ')
        #Not sure if this is the right approach
    text = text.replace('\n\n','\n') #Some pdfs have multple line breaks together. This helps a bit 
    chunks = character_splitter.split_text(text)
    # Removing stopwords
    filtered_chunks = []
    for chunk in chunks:
        filtered_chunks.append(filter_stopwords(stop_words,chunk))
    print('Creating embeddings...')
    embedded_docs = embeddings_model.encode(filtered_chunks)
    print('Upserting embeddings...')
    # fill_index(embedded_docs,chunks,pc_API_key,doc_name)
    if fill_index(embedded_docs,chunks,pc_API_key,doc_name):
        print(f'Embeddings created successfully for "{file_name}".')
    else: 
        print('Error creating embeddings for ' + doc_name)
else:
    print('-'*25)
    print(f'Embeddings for "{file_name}" already exist.')
if __name__ == '__main__': 
    query = ''
    context = ''
    number_of_tries = 0
    print('\n'+'-'*25)
    previous_answer = None #To start the loop without a history
    previous_question = None 
    n_substitution_chunks = 3 # Returns the "n" most relevant chunks when there are no matches
    while True: #Main Loop
        context = ''
        print('Type "Exit" or "Quit" to leave this chat.' )
        query = input('>>>Your question: ')
        if query in ['exit','EXIT','Exit', 'Quit', 'quit', 'q', 'QUIT', 'Q']: 
            break

        query_results = semantic_search(query,new_index,re_rank= True,threshold=metric_threshold,top_n=3,doc_name='sb_test')
        # print(query_results)
        if chunks_only:
            show_context(query_results,0,'yes',cross_scores=True) 
            query_results = semantic_search(query,new_index,re_rank= False,threshold=metric_threshold,top_n=3,doc_name='sb_test')
            print('WITHOUT RERANK: ','\n---------------------------------------')
            # MAYBE "SHOW MORE PASSAGES?"
            show_context(query_results,0,'yes',cross_scores=False) 
            # show_context(query_results,0,cross_scores=Falses
            continue
        response = get_completions_from_eden(query_results['documents'],query,2)
        print("AI's interpretation: ",response)
        print('*-----------------------------')
        print(f'Total costs: {response["cost"]}')
        print('*-----------------------------')
        time.sleep(5)
        show_context(query_results,2,'yes')
    print('Conversation ended successfully. See you soon!')