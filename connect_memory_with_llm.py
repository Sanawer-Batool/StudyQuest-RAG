import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,  # Pass token directly to constructor
        temperature=0.5,
        max_new_tokens=512  # Pass max_new_tokens as top-level parameter
    )
    return llm

# Step 2: Define Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Don't provide anything outside of the given context.

Context: {context}
Question: {question}

Answer: 
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Step 3: Load FAISS Database and Create Retrieval Chain
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create retriever
retriever = db.as_retriever(search_kwargs={'k': 3})

# Build the retrieval chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

llm = load_llm(HUGGINGFACE_REPO_ID)
prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

# Create the chain
qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Step 4: Invoke with a single query
user_query = input("Write Query Here: ")
response = qa_chain.invoke(user_query)
print("RESULT: ", response)

# Step 5: Retrieve source documents (optional, for debugging)
source_docs = retriever.invoke(user_query)
print("SOURCE DOCUMENTS: ", [doc.page_content for doc in source_docs])