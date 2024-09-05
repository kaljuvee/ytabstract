import streamlit as st
import os
import yaml
from dotenv import load_dotenv
from langchain.document_loaders import YoutubeLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load sample videos from YAML file
def load_sample_videos():
    with open("data/videos.yaml", "r") as file:
        return yaml.safe_load(file)

sample_videos = load_sample_videos()

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
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

    qa_yt = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=vectorstore.as_retriever())

    return qa_yt.run(query)

# Streamlit app
st.title("YouTube Video Summarizer")

# Dropdown for sample videos
selected_sample = st.selectbox(
    "Choose a sample video or enter your own URL below:",
    [""] + list(sample_videos.keys())
)

# YouTube URL input
if selected_sample:
    video_url = sample_videos[selected_sample]
    st.text(f"Selected video URL: {video_url}")
else:
    video_url = st.text_input("Enter YouTube Video URL:")

# Query input with default value
default_query = "Summarize the main points of this video"
query = st.text_input("Enter your query:", value=default_query)

if st.button("Summarize"):
    if video_url:
        if openai_api_key:
            with st.spinner("Processing video..."):
                summary = summarize_video(video_url, query)
            st.markdown(summary)
        else:
            st.error("OpenAI API key not found. Please check your .env file.")
    else:
        st.error("Please select a sample video or enter a valid YouTube URL.")