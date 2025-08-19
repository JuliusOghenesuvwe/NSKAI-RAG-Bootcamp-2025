import streamlit as st
import os
import re
import ast
from dotenv import load_dotenv
from helpers.youtubeloader import load_youtube_transcript
from helpers.chunker import chunk_documents
from helpers.vectorstore import create_vectorstore
from helpers.retriever import create_retriever
from helpers.chain import create_rag_chain, generate_suggested_questions_chain, create_summarization_chain

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="YouTube Q&A Chatbot",
    page_icon="💬",
    layout="wide"
)

# CSS for alignment and spacing
st.markdown(r"""
    <style>
           .block-container {
                padding-top: 2rem;
            }
           [data-testid="stSidebar"] > div:first-child {
                padding-top: 2rem;
           }
           h1, h2, h3 {
                margin-top: 0;
                padding-top: 0;
           }
    </style>
    """, unsafe_allow_html=True)

st.title("YouTube Video Q&A with RAG 💬")
st.markdown("Ask questions about any YouTube video using AI-powered retrieval!")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "youtube_url" not in st.session_state:
    st.session_state.youtube_url = ""
if "processed_url" not in st.session_state:
    st.session_state.processed_url = ""
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.1
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "prompt_from_suggestion" not in st.session_state:
    st.session_state.prompt_from_suggestion = None


def is_valid_youtube_url(url):
    """Check if the URL is a valid YouTube URL."""
    pattern = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    return re.match(pattern, url)

def process_video():
    """Process the YouTube video."""
    if st.session_state.youtube_url and is_valid_youtube_url(st.session_state.youtube_url):
        if st.session_state.youtube_url != st.session_state.processed_url:
            with st.spinner("Processing..."):
                try:
                    documents = load_youtube_transcript(st.session_state.youtube_url)
                    chunks = chunk_documents(documents)
                    st.session_state.chunks = chunks
                    vectorstore = create_vectorstore(chunks)
                    st.session_state.vectorstore = vectorstore
                    retriever = create_retriever(vectorstore)
                    st.session_state.retriever = retriever
                    chain = create_rag_chain(retriever, st.session_state.temperature)
                    st.session_state.chain = chain
                    st.session_state.processed_url = st.session_state.youtube_url
                    st.session_state.messages = []
                    generate_suggested_questions(chunks)
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
    elif st.session_state.youtube_url:
        st.error("Please enter a valid YouTube URL.")

def generate_suggested_questions(chunks):
    """Generate suggested questions from the video transcript and ensure there are always 4."""
    if chunks:
        transcript_summary = " ".join([chunk.page_content for chunk in chunks[:2]]).strip()
        if len(transcript_summary) > 2000:
            transcript_summary = transcript_summary[:2000]
        
        generated_questions = []
        try:
            question_chain = generate_suggested_questions_chain(st.session_state.temperature)
            response = question_chain.invoke({"transcript_summary": transcript_summary})
            generated_questions = ast.literal_eval(response)
            if not isinstance(generated_questions, list):
                generated_questions = []
        except (ValueError, SyntaxError, Exception):
            generated_questions = []

        default_questions = [
            "What is the main topic of the video?",
            "Can you summarize the key points?",
            "Who is the intended audience for this video?",
            "What are the main takeaways from this video?"
        ]
        
        unique_defaults = [q for q in default_questions if q not in generated_questions]
        
        final_questions = generated_questions + unique_defaults
        st.session_state.suggested_questions = final_questions[:4]

def handle_question(question):
    """Handle user question submission."""
    st.session_state.messages.append({"role": "user", "content": question})
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chain({"question": question})
                answer = response.get("answer", "I couldn't generate an answer.")
                context = response.get("context", "")
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer, "context": context})
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# --- UI LAYOUT --- #

# Sidebar
with st.sidebar:
    if st.session_state.get("processed_url"):
        st.header("✅ Transcript Loaded")
        if st.button("Process another video"):
            st.session_state.youtube_url = ""
            st.session_state.processed_url = ""
            st.session_state.messages = []
            st.session_state.suggested_questions = []
            st.session_state.chain = None
            st.session_state.vectorstore = None
            st.session_state.retriever = None
            st.session_state.chunks = []
            st.session_state.prompt_from_suggestion = None
            st.rerun()
    else:
        st.header("🔗 Paste YouTube link below")
        st.text_input("YouTube URL", key="youtube_url", on_change=process_video, label_visibility="collapsed")

    st.header("⚙️ Settings")
    st.session_state.temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=st.session_state.temperature, step=0.1)
    st.caption("Lower values make the AI more factual and less random.")
    
    if st.session_state.chain and st.button("📝 Summarize Video"):
        if st.session_state.chunks:
            with st.spinner("Summarizing..."):
                summarize_chain = create_summarization_chain(st.session_state.temperature)
                full_transcript = " ".join([chunk.page_content for chunk in st.session_state.chunks])
                max_length = 12000
                if len(full_transcript) > max_length:
                    full_transcript = full_transcript[:max_length]
                summary = summarize_chain.invoke({"transcript": full_transcript})
                st.session_state.messages.append({"role": "assistant", "content": f"**Video Summary:**\n\n{summary}"})
                st.rerun()
        else:
            st.warning("Please process a video first.")


# Main layout
if not st.session_state.chain:
    st.info("👈 Please process a YouTube video first using the sidebar.")
else:
    col1, col2 = st.columns([2, 1])

    with col1:
        with st.container(border=True):
            st.header("Chat")
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        if message["role"] == "assistant" and "context" in message and message["context"]:
                            with st.expander("Show Source Context"):
                                st.info(message["context"])
            
            prompt = st.chat_input("Ask a question about the video...")
            if st.session_state.get("prompt_from_suggestion"):
                prompt = st.session_state.prompt_from_suggestion
                st.session_state.prompt_from_suggestion = None

            if prompt:
                handle_question(prompt)
                st.rerun()

    with col2:
        with st.container(border=True):
            st.header("Suggested Questions")
            if st.session_state.suggested_questions:
                for question in st.session_state.suggested_questions:
                    if st.button(question, key=f"suggested_{question}"):
                        st.session_state.prompt_from_suggestion = question
                        st.rerun()
            else:
                st.info("No suggested questions available yet.")