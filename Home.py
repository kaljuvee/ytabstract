import streamlit as st
import os
from dotenv import load_dotenv
from langchain.document_loaders import YoutubeLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

# Set your OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

def get_youtube_id(url):
    # Extract video ID from YouTube URL
    if "youtu.be" in url:
        return url.split("/")[-1]
    elif "youtube.com" in url:
        return url.split("v=")[1].split("&")[0]
    else:
        return url

def summarize_video(video_url, query):
    video_id = get_youtube_id(video_url)
    
    # Load documents with YoutubeLoader
    loader = YoutubeLoader(video_id=video_id, language="en")
    yt_docs = loader.load_and_split()
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(yt_docs, embeddings)

    # Define LLM
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

    qa_yt = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=vectorstore.as_retriever())

    return qa_yt.run(query)

# Streamlit app
st.title("YouTube Video Summarizer")

# YouTube URL input
video_url = st.text_input("Enter YouTube Video URL:")

# Query input with default value
default_query = "Summarize the main points of this video"
query = st.text_input("Enter your query:", value=default_query)

if st.button("Summarize"):
    if video_url:
        with st.spinner("Processing video..."):
            summary = summarize_video(video_url, query)
        st.markdown(summary)
    else:
        st.error("Please enter a valid YouTube URL.")