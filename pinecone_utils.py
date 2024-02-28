from pinecone import Pinecone,PodSpec
import itertools

def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


def pinecone_index(api_key,index_name = 'smart-bible'):
    pc = Pinecone(api_key=api_key)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='cosine',
            spec= PodSpec(environment="us-west1-gcp", pod_type="p1.x1")
        )
        print(f'{index_name} created successfully!')
    return pc.Index(index_name)

def fill_index(embeddings,texts,api_key,doc_name = 'smart_bible',index_name ='smart-bible'):
    pc = Pinecone(api_key=api_key)
    docs = [
        {
            'id': f'{doc_name}-child_doc{n+1}',
            'values': e,
            'metadata': {'text': t},
         } for n,(e,t) in enumerate(zip(embeddings,texts))
    ]
    try:
        with pc.Index(index_name, pool_threads=len(docs)) as index:
        # Send requests in parallel
            async_results = [
                index.upsert(
                    vectors=c,
                    namespace = doc_name,
                    async_req=True
                    )
                for c in chunks(docs, batch_size=100)
            ]
            # Wait for and retrieve responses (this raises in case of error)
            [async_result.get() for async_result in async_results]
        return True
    except:
        return False
def docs_exist(index):
    res = index.describe_index_stats()
    if res['total_vector_count'] >20: #Should get a better solution later!
        return True
    return False

### Pinecone Setup







