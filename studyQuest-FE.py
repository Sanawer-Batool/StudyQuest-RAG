import os
import streamlit as st
import re
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.docstore.document import Document
import tempfile
import base64

# Custom CSS for stunning theme
st.markdown("""
<style>
    .stApp {
        background-color: #1e2a44;
        color: #ffffff;
    }
    .main {
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .stButton>button {
        background-color: #00c4cc;
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #009aa6;
        transform: scale(1.05);
    }
    .stFileUploader {
        background-color: rgba(255, 255, 255, 0.95);
        border: 2px dashed #00c4cc;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .stFileUploader label {
        color: #00c4cc !important;
        font-weight: bold;
    }
    .stTextInput {
        background-color: rgba(255, 255, 255, 0.95);
        border: 1px solid #00c4cc;
        border-radius: 10px;
        padding: 5px 10px;
        color: #1e2a44;
    }
    .stTextInput label {
        color: #00c4cc !important;
        font-weight: bold;
    }
    .chat-message-user {
        background-color: #2e4a7d;
        color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .chat-message-assistant {
        background-color: #3a5e8c;
        color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background-color: rgba(255, 255, 255, 0.95);
        border-right: 1px solid #00c4cc;
        padding: 20px;
        border-radius: 10px;
    }
    .header {
        text-align: center;
        color: #00c4cc;
        font-size: 3em;
        font-weight: bold;
        background: linear-gradient(45deg, #1e2a44, #2e4a7d);
        padding: 10px;
        border-radius: 10px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    .instructions {
        background-color: rgba(255, 255, 255, 0.95);
        color: #1e2a44;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 4px solid #00c4cc;
    }
    .stExpander {
        background-color: #2e4a7d;
        border: 1px solid #00c4cc;
        border-radius: 10px;
    }
    .stExpander > div > div {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        max_new_tokens=512
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

# Step 3: Create Vector Store from Uploaded PDF
def get_vectorstore_from_pdf(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embedding_model)

        os.unlink(tmp_file_path)
        return vectorstore
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

# Step 4: Create Vector Store from YouTube Transcript
def get_vectorstore_from_youtube(video_id):
    st.write(f"Processing YouTube video ID: {video_id}")  # Debug output
    if not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
        st.error("Invalid YouTube video ID. It should be 11 characters long (e.g., Gfr50f6ZBvo).")
        return None
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        documents = [Document(page_content=transcript, metadata={"source": f"YouTube Video ID: {video_id}"})]
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        docs = splitter.split_documents(documents)

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embedding_model)
        return vectorstore
    except TranscriptsDisabled:
        st.error("No captions available for this video.")
        return None
    except Exception as e:
        st.error(f"Error processing YouTube transcript: {str(e)}")
        return None

# Step 5: Merge Vector Stores
def merge_vectorstores(pdf_vectorstore, youtube_vectorstore):
    if pdf_vectorstore and youtube_vectorstore:
        pdf_vectorstore.merge_from(youtube_vectorstore)
        return pdf_vectorstore
    elif pdf_vectorstore:
        return pdf_vectorstore
    elif youtube_vectorstore:
        return youtube_vectorstore
    return None

# Step 6: Streamlit App
def main():
    st.markdown('<div class="header">StudyQuest Chatbot</div>', unsafe_allow_html=True)
    
    header_image_container = st.container()
    with header_image_container:
        try:
            with open("notion cover.jpg", "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode()
                st.image(f"data:image/jpeg;base64,{encoded_image}", width=800)
        except FileNotFoundError:
            st.image("https://via.placeholder.com/800x200.png?text=StudyQuest+Header", width=800, caption="StudyQuest Header")

    # Sidebar for PDF upload and YouTube video ID
    with st.sidebar:
        st.markdown('<h3 style="color: #00c4cc; text-align: center;">Upload Your Study Material</h3>', unsafe_allow_html=True)
        st.markdown("""
        <div class="instructions">
        Upload a PDF or enter a YouTube video ID!
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize session state for inputs
        if 'uploaded_file' not in st.session_state:
            st.session_state.uploaded_file = None
        if 'youtube_id' not in st.session_state:
            st.session_state.youtube_id = ""

        uploaded_file = st.file_uploader("Choose a PDF file:", type=["pdf"], key="pdf_uploader")
        youtube_id = st.text_input("Enter YouTube Video ID:", key="youtube_id_input", value=st.session_state.youtube_id)

        # Clear All button
        if st.button("Clear All"):
            st.session_state.vectorstore = None
            st.session_state.uploaded_file = None
            st.session_state.youtube_id = ""
            st.session_state.messages = []
            st.rerun()

        # Update session state with current inputs
        st.session_state.uploaded_file = uploaded_file
        st.session_state.youtube_id = youtube_id

        # Display current sources
        current_sources = []
        if st.session_state.uploaded_file:
            current_sources.append("PDF")
        if st.session_state.youtube_id:
            current_sources.append(f"YouTube Video ID: {st.session_state.youtube_id}")
        if current_sources:
            st.markdown(f"**Current Sources:** {', '.join(current_sources)}")
        else:
            st.markdown("**Current Sources:** None")

        if st.button("Process Materials"):
            if not st.session_state.uploaded_file and not st.session_state.youtube_id:
                st.error("Please upload a PDF or enter a YouTube video ID.")
            else:
                with st.spinner("Processing your materials..."):
                    # Reset vector store
                    st.session_state.vectorstore = None
                    pdf_vectorstore = get_vectorstore_from_pdf(st.session_state.uploaded_file) if st.session_state.uploaded_file else None
                    youtube_vectorstore = get_vectorstore_from_youtube(st.session_state.youtube_id) if st.session_state.youtube_id else None
                    st.session_state.vectorstore = merge_vectorstores(pdf_vectorstore, youtube_vectorstore)
                
                if st.session_state.vectorstore:
                    st.success("Materials processed successfully! Start asking questions.")
                else:
                    st.error("Failed to process materials. Please check your inputs.")

    # Initialize session state for messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    # Main content area
    with st.container():
        for message in st.session_state.messages:
            with st.chat_message(message['role'], avatar="user" if message['role'] == 'user' else "assistant"):
                st.markdown(f'<div class="chat-message-{message["role"]}">{message["content"]}</div>', unsafe_allow_html=True)

        prompt = st.chat_input("Ask a question about your study material", disabled=not st.session_state.vectorstore)

        if prompt and st.session_state.vectorstore:
            with st.chat_message('user', avatar="user"):
                st.markdown(f'<div class="chat-message-user">{prompt}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({'role': 'user', 'content': prompt})

            try:
                retriever = st.session_state.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                llm = load_llm(HUGGINGFACE_REPO_ID)
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

                with st.spinner("Crafting your answer..."):
                    result = qa_chain.invoke(prompt)

                source_docs = retriever.invoke(prompt)
                source_docs_text = [doc.page_content for doc in source_docs]
                result_to_show = f"**Answer:** {result}\n\n**Source Documents:**\n{source_docs_text}"

                with st.chat_message('assistant', avatar="assistant"):
                    st.markdown(f'<div class="chat-message-assistant">{result}</div>', unsafe_allow_html=True)
                    with st.expander("View Source Documents"):
                        for i, doc in enumerate(source_docs_text, 1):
                            st.markdown(f"**Document {i}:**\n{doc}")
                st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

            except Exception as e:
                st.error(f"Error: {str(e)}")

        elif prompt and not st.session_state.vectorstore:
            st.error("Please upload a PDF or enter a YouTube video ID and process the materials first.")

if __name__ == "__main__":
    main()
