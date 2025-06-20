import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Step 1: Setup Vector Store
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

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

# Step 3: Setup LLM (Mistral with HuggingFace)
def load_llm(huggingface_repo_id, hf_token):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=hf_token,  # Pass token directly to constructor
        temperature=0.5,
        max_new_tokens=512  # Pass max_new_tokens as top-level parameter
    )
    return llm

# Step 4: Streamlit App
def main():
    st.title("Ask Chatbot!")

    # Initialize session state for messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Get user input
    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Setup LLM and chain
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            # Load vector store
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            # Create retriever
            retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

            # Build the retrieval chain
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)
            prompt_template = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

            qa_chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | prompt_template
                | llm
                | StrOutputParser()
            )

            # Invoke chain
            result = qa_chain.invoke(prompt)

            # Get source documents for display
            source_docs = retriever.invoke(prompt)
            source_docs_text = [doc.page_content for doc in source_docs]
            result_to_show = f"{result}\n\nSource Docs:\n{source_docs_text}"

            # Display response
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()