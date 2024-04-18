# RAG-QA-Generator

# PDF Question Generator

PDF Question Generator is a Streamlit application that generates various types of questions from a PDF document. It leverages language models and natural language processing techniques to extract relevant information from the PDF and create questions like Multiple Choice Questions (MCQs), True/False questions, and long questions with one-word answers.

## Features

- **Multiple Choice Questions (MCQs)**: Generates a set of MCQs along with their answers.
- **True/False Questions**: Generates a set of True/False questions along with their answers.
- **Long Questions with One-word Answers**: Generates long questions where the answers are expected to be a single word.
- **File Upload**: Allows users to upload PDF. The questions generated from the PDF file are saved as TXT.

## Technologies Used

- **Language Model**: llama 2
- **Document Loader**: PyPDFLoader
- **Text Splitter**: RecursiveCharacterTextSplitter
- **Embeddings**: HuggingFaceEmbeddings
- **Question Answering**: RetrievalQA
- **Vector Store**: FAISS
- **Frontend**: Streamlit

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
2. Upload a PDF file using the file uploader.
3. View the generated questions in the Streamlit app interface.
4. The generated questions will also be saved in a Generated_Questions.txt file in the same directory.
5. Users can also upload a TXT file for processing.

## Sample Output:
![Screenshot (39)](https://github.com/jatiink/RAG-QA-Generator/assets/97089717/dd13fbed-aff6-4032-bcc5-5bca781cd0f1)
