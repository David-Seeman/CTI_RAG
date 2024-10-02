import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch

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

def main():
    # Step 1: Get the webpage URL from the user
    url = input("Enter the URL of the webpage: ").strip()
    if not url:
        print("No URL provided. Exiting.")
        return

    # Step 2: Fetch and clean the webpage content
    print("\nFetching webpage content...")
    try:
        text = fetch_webpage_content(url)
        print("Webpage content fetched successfully.\n")
    except Exception as e:
        print(f"Error: {e}")
        return

    # Step 3: Split the text into chunks for processing
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    print(f"Text split into {len(chunks)} chunks.\n")

    # Step 4: Create embeddings for the text chunks
    print("Creating embeddings for text chunks...")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_store = FAISS.from_texts(chunks, embeddings)
    print("Embeddings created and vector store initialized.\n")

    # Step 5: Load the language model
    print("Loading the language model...")
    model_name = "google/flan-t5-small"  # You can choose other models like 't5-base' or 't5-large'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
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
        return_source_documents=False
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