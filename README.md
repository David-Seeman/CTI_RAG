#Virtual Environment Commands for Mac

python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip

pip install requests beautifulsoup4 langchain faiss-cpu transformers torch langchain-community sentence-transformers pymupdf spacy

python -m spacy download en

pip install 'numpy<2.0.0' 

#FOR AYHAM .ipynb FILE: 

pip install langchain_openai
