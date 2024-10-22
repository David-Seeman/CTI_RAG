import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
import torch
import pymupdf
import os
from summarizer import Summarizer

def fetch_webpage_content(url):
    """
    Fetches and cleans text content from the specified URL.
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve the webpage: {url}")
    soup = BeautifulSoup(response.text, 'html.parser')
    # Remove script and style elements
    for element in soup(['script', 'style']):
        element.decompose()
    text = soup.get_text(separator='\n')
    # Clean up the text
    lines = [line.strip() for line in text.splitlines()]
    cleaned_text = '\n'.join(line for line in lines if line)
    return cleaned_text

def extract_text_from_pdf(pdf_path):
    """
    Extracts text content from a PDF file.
    """
    text = ""
    with pymupdf.open(pdf_path) as pdf:
        for page_num in range(len(pdf)):
            page = pdf.load_page(page_num)
            text += page.get_text()
    return text

def read_file_content(file_path):
    """
    Reads and returns the content of a text or PDF file.
    """
    if not os.path.exists(file_path):
        raise Exception(f"File not found: {file_path}")

    if file_path.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
         
def summarize_chunk(chunk):
    """
    Generates a brief summary for a given chunk.
    Testing semantic anchor using a BERT-based summarizer.
    """

    model = Summarizer()
    summary = model.get_summary(chunk, "randomTitle")
    full = summary[0]['sentence']
    # print("Summary: " + summary)
    return summary

def main():
    # Step 1: Get the webpage URL or file path from the user
    user_input = input("Enter the URL of the webpage or the path to a text or PDF file: ").strip()
    if not user_input:
        print("No input provided. Exiting.")
        return

    # Step 2: Fetch and clean the webpage content
    print("\nFetching content...")
    try:
        if user_input.startswith("http://") or user_input.startswith("https://"):
            # If input is a URL, fetch webpage content
            text = fetch_webpage_content(user_input)
            print("Webpage content fetched successfully.\n")
        else:
            # Otherwise, treat input as a file path
            text = read_file_content(user_input)
            print("File content read successfully.\n")
    except Exception as e:
        print(f"Error: {e}")
        return

    # Step 3: Split the text into chunks for processing
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    print(f"Text split into {len(chunks)} chunks.\n")

    # Create a list of dictionaries to store chunks with their metadata
    chunk_metadata = []
    for idx, chunk in enumerate(chunks):
        metadata = {
            "index": idx,
            "summary": summarize_chunk(chunk),
            "length": len(chunk)
        }
        chunk_metadata.append((chunk, metadata))
    print(f"Text split into {len(chunks)} chunks with metadata.\n")

    # Step 4: Create embeddings for the text chunks
    print("Creating embeddings for text chunks...")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_store = FAISS.from_texts(
        [chunk for chunk, _ in chunk_metadata], 
        embeddings, 
        metadatas=[meta for _, meta in chunk_metadata]
    )
    print("Embeddings created and vector store initialized.\n")

    # Step 5: Load the language model
    print("Loading the language model...")
    model_name = "Qwen/Qwen2.5-7B-Instruct"  # You can choose other models like 't5-base' or 't5-large'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    print("Language model loaded successfully.\n")

    # Step 6: Initialize the RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=False,
    )

    # Step 7: Interactive Q&A Loop
    print("You can now ask questions about the webpage content.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Your Question: ").strip()
        if query.lower() in ['exit', 'quit']:
            print("Exiting the program. Goodbye!")
            break
        if not query:
            print("Please enter a valid question.\n")
            continue
        print("\nProcessing your question...")
        try:
            answer = qa.run(query)
            print(f"\nAnswer: {answer}\n")
        except Exception as e:
            print(f"Error generating answer: {e}\n")

if __name__ == "__main__":
    main()