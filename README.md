# StudyQuest-RAG
StudyQuest-RAG is a Streamlit-based chatbot that leverages Retrieval-Augmented Generation (RAG) to answer questions based on uploaded PDF study notes or YouTube video transcripts. Built with LangChain, FAISS, and HuggingFace models, it ensures answers are grounded in the provided materials, avoiding hallucinations.
Features

PDF Processing: Upload PDF study notes, which are split, embedded, and stored in a FAISS vector store for retrieval.
YouTube Transcripts: Enter a YouTube video ID to process its English transcript, enabling question-answering based on video content.
Concurrent Input Support: Process both PDF and YouTube inputs simultaneously, merging their vector stores for unified retrieval.
Customizable UI: Sleek, dark-themed interface with custom CSS for enhanced user experience.
No Hallucinations: Answers are strictly based on the provided context, powered by a Mistral-7B model.
Clear Session Management: Includes a "Clear All" button to reset inputs and chat history.

## Tech Stack

Frontend: Streamlit
Backend: LangChain, FAISS, HuggingFace (Mistral-7B-Instruct-v0.3, sentence-transformers/all-MiniLM-L6-v2)
APIs: YouTube Transcript API
Python Libraries: streamlit, langchain-huggingface, langchain-community, faiss-cpu, youtube-transcript-api, PyPDF2, tiktoken

## Prerequisites

Python 3.8+
A HuggingFace API token (set as HF_TOKEN environment variable)
Git installed for cloning the repository
A YouTube video with English captions for transcript processing

## Setup Instructions

Clone the Repository:
git clone https://github.com/Sanawer-Batool/StudyQuest-RAG.git
cd StudyQuest-RAG


Create a Virtual Environment:(on Windows)
python -m venv venv
venv\Scripts\activate


### Install Dependencies:
pip install -r requirements.txt

If requirements.txt is not present, install the following:
pip install streamlit langchain-huggingface langchain-community faiss-cpu youtube-transcript-api PyPDF2 tiktoken sentence-transformers


### Set Environment Variable:

Create a .env file in the root directory:HF_TOKEN=your_huggingface_api_token


Or export it directly:export HF_TOKEN=your_huggingface_api_token  # On Windows: set HF_TOKEN=your_huggingface_api_token




### Run the Application:
streamlit run app.py



## Usage

Launch the App:
Open your browser to the URL provided by Streamlit (typically http://localhost:8501).


## Upload Materials:
In the sidebar, upload a PDF study note or enter a YouTube video ID (e.g., Gfr50f6ZBvo).
Click Process Materials to index the content.


## Ask Questions:
Use the chat input to ask questions about the processed materials.
Answers are displayed with source documents in an expandable section.


## Clear Inputs:
Click Clear All to reset the PDF, YouTube ID, and chat history.


Current Sources:
The sidebar shows which sources (PDF, YouTube, or both) are currently loaded.


Example

PDF: Upload a lecture note PDF on machine learning.
YouTube: Enter a video ID for a tutorial on neural networks.
Question: "What is backpropagation?"
Result: The app retrieves relevant chunks from both sources and provides a concise answer.

## Issues

If the YouTube Transcript API fails, ensure the video has English captions.
For PDF processing errors, verify the file is not corrupted.
Report bugs or feature requests via GitHub Issues.


# Developed by Sanawer Batool
