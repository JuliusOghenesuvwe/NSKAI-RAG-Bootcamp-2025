class SimpleVectorStore:
    def __init__(self, documents):
        self.documents = documents
    
    def as_retriever(self, **kwargs):
        return SimpleRetriever(self.documents)

class SimpleRetriever:
    def __init__(self, documents):
        self.documents = documents
    
    def get_relevant_documents(self, query):
        return self.documents[:3]  # Return first 3 docs

def create_vectorstore(documents):
    return SimpleVectorStore(documents)