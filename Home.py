import streamlit as st
import yaml
import os
from dotenv import load_dotenv
from langchain.document_loaders import YoutubeLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Load sample videos from YAML file with UTF-8 encoding
def load_sample_videos():
    with open("data/videos.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

sample_videos = load_sample_videos()

def get_youtube_id(url):
    # Extract video ID from YouTube URL
    if "youtu.be" in url:
        return url.split("/")[-1]
    elif "youtube.com" in url and "v=" in url:
        v_index = url.index("v=")
        amp_index = url.index("&") if "&" in url else len(url)
        return url[v_index+2:amp_index]
    else:
        return url

def summarize_video(video_url, query, model_name, api_key):
    video_id = get_youtube_id(video_url)
    
    # Load documents with YoutubeLoader
    loader = YoutubeLoader(video_id=video_id, language="en")
    yt_docs = loader.load_and_split()
    
    if not yt_docs:
        raise ValueError("No content could be loaded from the video.")
    
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(yt_docs, embeddings)
    # Define LLM with the selected model
    llm = ChatOpenAI(model_name=model_name, temperature=0.7, openai_api_key=api_key)
    qa_yt = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=vectorstore.as_retriever())
    return qa_yt.run(query)

# Streamlit app
st.title("YouTube Video Abstracts")

# Sidebar for API key
st.sidebar.title("Settings")
user_api_key = st.sidebar.text_input("Enter your OpenAI API key (https://platform.openai.com/api-keys)", type="password")
save_key = st.sidebar.button("Save API Key")
if save_key and user_api_key:
    st.sidebar.success("API Key saved successfully!")

# Get API key from user input or environment variable
api_key = user_api_key or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.sidebar.warning("No API key found. Please enter a key or set the OPENAI_API_KEY environment variable.")

# Main content
# Dropdown for sample videos
selected_sample = st.selectbox(
    "Choose a sample video or enter your own URL below:",
    [""] + list(sample_videos.keys())
)

# YouTube URL input
custom_url = st.text_input("Enter YouTube Video URL:", "")

# Determine which URL to use
if custom_url:
    video_url = custom_url
elif selected_sample:
    video_url = sample_videos[selected_sample]
else:
    video_url = ""

# Display the selected or entered URL
if video_url:
    st.text(f"Selected video URL: {video_url}")

# Model selection dropdown
model_options = ["gpt-4o", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
selected_model = st.selectbox("Choose a model:", model_options, index=0)

# Query input with default value
default_query = "Summarize the main points of this video, reply in bullet point format."
query = st.text_input("Enter your query:", value=default_query)

if st.button("Summarize"):
    if video_url:
        if api_key:
            with st.spinner(f"Processing video using {selected_model}..."):
                try:
                    summary = summarize_video(video_url, query, selected_model, api_key)
                    st.markdown(summary)
                except ValueError as ve:
                    st.error(f"Error: {str(ve)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
        else:
            st.error("No API key available. Please enter your OpenAI API key in the sidebar or set it as an environment variable.")
    else:
        st.error("Please select a sample video or enter a valid YouTube URL.")
