import streamlit as st
from langchain.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

# Initialize Ollama model
llm = Ollama(model='llama2', temperature=0.3)

# Function to generate questions
def generate_questions(text):
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    # Split the document into chunks
    texts = text_splitter.split_documents(text)

    # Initialize embedding model
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': False}
    )

    # Create vector database
    vectordb = FAISS.from_documents(texts, embedding=embedding)
    retriever = vectordb.as_retriever()

    # Initialize RetrievalQA model
    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type='stuff',
                                     retriever=retriever,
                                     verbose=True)

    # Dictionary to store generated questions
    questions = {}

    # Queries to generate questions
    queries = ["Provide 10 MCQ questions with answers",
               "Provide 10 true/false questions with answers",
               "Provide 10 long questions that have answers only in one word with answers"]

    # Generate questions for each query
    for query in queries:
        result = qa.run(query)
        questions[query] = result

    return questions

# Function to save questions to txt file
def save_to_txt(questions):
    # Open txt file to save questions
    with open("Generated_Questions.txt", "w") as f:
        # Write each query and its result to the file
        for query, result in questions.items():
            f.write(query + "\n")
            f.write(str(result) + "\n\n")

# Set gradient background
background_gradient = """
<style>
body {
    background-image: linear-gradient(120deg, #f6d365, #fda085);
}
</style>
"""
st.markdown(background_gradient, unsafe_allow_html=True)

# Streamlit app
def main():
    # Set app title
    st.title("PDF Question Generator")
    
    # Display file uploader
    st.markdown("<h3 style='text-align: left;'>Upload a PDF file</h3>", unsafe_allow_html=True)

    # File uploader component
    uploaded_file = st.file_uploader("", type="pdf", key="file_uploader", accept_multiple_files=False)

    if uploaded_file is not None:
        # Save uploaded PDF file
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load PDF file
        file_path = uploaded_file.name
        loader = PyPDFLoader(file_path)
        text = loader.load()

        # Generate questions
        questions = generate_questions(text)

        # Save questions to txt file
        save_to_txt(questions)

        # Display generated questions
        for query, result in questions.items():
            st.subheader(query)
            st.write(result)

if __name__ == "__main__":
    main()
