# YouTube Video Q&A with RAG 💬

This is an interactive Q&A application that allows you to "chat" with any YouTube video. It uses a Retrieval-Augmented Generation (RAG) pipeline to understand the video's content and answer your questions based on the transcript.

## Features

- **Interactive Q&A:** Ask questions about a YouTube video in natural language.
- **Fast Generation:** Powered by the incredibly fast **Groq** API with Llama 3 for near-instant answers.
- **Simple UI:** Built with **Streamlit** for a clean and easy-to-use web interface.

## Tech Stack

- **Framework:** LangChain (for prompts and LLM)
- **UI:** Streamlit
- **LLM:** Groq (Llama 3 8B)
- **Vector Store:** Simple custom retriever (placeholder)
- **Data Loader:** `yt-dlp`, `youtube-transcript-api`

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/youtube-qa-chatbot.git
cd youtube-qa-chatbot
```

### 2. Create a Python Virtual Environment

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `streamlit` - Web UI framework
- `langchain` - RAG framework components
- `langchain-groq` - Groq LLM integration
- `langchain-community` - Community integrations
- `yt-dlp` - YouTube video downloader
- `python-dotenv` - Environment variable management
- `chromadb` - Vector database (not currently used)
- `youtube-transcript-api` - YouTube transcript extraction

### 4. Set Up Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Get your API key from the [GroqCloud Console](https://console.groq.com/keys).

3. Open the `.env` file and add your API key:
   ```
   GROQ_API_KEY="your_groq_api_key_here"
   ```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

1. Paste a YouTube video URL into the sidebar input field.
2. Click the **"Process Video"** button and wait for the processing to complete.
3. Once processed, you can ask questions about the video in the main input field.

## 🚀 Deploy

### Streamlit Cloud (Free)

To make your app accessible to others:

1. **Visit** https://share.streamlit.io
2. **Sign in** with GitHub
3. **Click "New app"**
4. **Select your repository**
5. **Set main file:** `app.py`
6. **Add secrets:** Click "Advanced settings" → "Secrets" → Add:
   ```
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
7. **Click "Deploy"**

### Render

This application can also be deployed on Render using the provided `render.yaml` and `Dockerfile`. You will need to set up a new "Web Service" on Render and connect it to your GitHub repository. Render will use the `render.yaml` file to configure the deployment. You will also need to add your `GROQ_API_KEY` as a secret in the Render dashboard.

## Project Structure

```
.
├── helpers/
│   ├── __init__.py       # Makes 'helpers' a Python package
│   ├── chain.py          # Creates the final RAG chain with the LLM
│   ├── chunker.py        # Splits documents into smaller chunks
│   ├── retriever.py      # Creates the retriever
│   ├── vectorstore.py    # Creates the vector store
│   └── youtubeloader.py  # Loads and cleans transcripts using yt-dlp
├── .env                  # Stores API keys (secret, not committed to git)
├── .env.example          # Example environment file
├── .gitignore            # Specifies files for git to ignore
├── app.py                # The main Streamlit application file
├── requirements.txt      # Project dependencies
├── Dockerfile            # Dockerfile for deployment
├── render.yaml           # Render deployment configuration
└── README.md             # This file
```

## How It Works

The application follows a simplified RAG pipeline:

1. **Ingestion:** The `youtubeloader` fetches the video transcript using `yt-dlp` and `youtube-transcript-api`.
2. **Chunking:** The `chunker` splits the clean transcript into smaller, overlapping documents.
3. **Indexing:** The `vectorstore` helper creates a simple in-memory store for the document chunks.
4. **Retrieval:** When a question is asked, the `retriever` returns the first few chunks of the transcript as context.
5. **Generation:** The retrieved chunks and the original question are passed to the Groq LLM within a structured prompt, which then generates the final, grounded answer.
