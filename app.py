from langchain.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

llm = Ollama(model='llama2',
             temperature=0.9)
print(llm("who is the president of united states?"))

loader = PyPDFLoader("Big Mac Index.pdf")
text = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function = len,)

texts = text_splitter.split_documents(text)

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-mpnet-base-v2",
    model_kwargs = {'device': 'cuda'},
    encode_kwargs = {'normalize_embeddings': False}
    )

vectordb = FAISS.from_documents(texts, embedding=embedding)
retriever = vectordb.as_retriever()

qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type='stuff',
                                 retriever=retriever,
                                 verbose=True)

def test_rag(qa, query):
    print(f"Query: {query}\n")
    result = qa.run(query)
    print("\nResult: ", result)

query = "provide True/False questions from the passage"
test_rag(qa, query)