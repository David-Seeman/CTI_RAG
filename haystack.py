from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, TransformersGenerator
from haystack.prompts import PromptTemplate

# Initialize the Document Store
document_store = InMemoryDocumentStore()

# Write documents to the store
documents = [
    Document(content="My name is Jean and I live in Paris."),
    Document(content="My name is Mark and I live in Berlin."),
    Document(content="My name is Giorgio and I live in Rome.")
]
document_store.write_documents(documents)

# Define the prompt template
prompt_template = """
Given these documents, answer the question.
Documents:
{% for doc in documents %}
- {{ doc.content }}
{% endfor %}
Question: {{ question }}
Answer:
"""

# Initialize the Retriever
retriever = BM25Retriever(document_store=document_store)

# Initialize the Generator with a local model
generator = TransformersGenerator(
    model_name_or_path="facebook/bart-large-cnn",  # Replace with your chosen model
    use_gpu=True  # Set to False if you don't have a GPU
)

# Initialize the Prompt Template
prompt = PromptTemplate(template=prompt_template)

# Create the RAG Pipeline
rag_pipeline = Pipeline()

# Add Retriever to the pipeline
rag_pipeline.add_node(
    component=retriever,
    name="Retriever",
    inputs=["Query"]
)

# Add Prompt Builder to the pipeline
rag_pipeline.add_node(
    component=prompt,
    name="PromptBuilder",
    inputs=["Retriever"]
)

# Add Generator to the pipeline
rag_pipeline.add_node(
    component=generator,
    name="Generator",
    inputs=["PromptBuilder"]
)

# Define the question
question = "Who lives in Paris?"

# Run the pipeline
results = rag_pipeline.run(
    query=question,
    params={
        "Retriever": {"top_k": 10},
        "Generator": {"max_length": 200}
    }
)

# Print the generated answer
print(results["Generator"])