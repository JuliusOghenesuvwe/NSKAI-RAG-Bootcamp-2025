from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os
import streamlit as st

def create_rag_chain(retriever, temperature=0.1):
    """Create the RAG chain with Groq LLM"""
    
    # Initialize Groq LLM
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama3-8b-8192",
        temperature=temperature
    )
    
    # Create prompt template
    template = """You are a helpful AI assistant that answers questions based on YouTube video transcripts.
    
    Context from the video:
    {context}
    
    Question: {question}
    
    Instructions:
    - Answer the question based only on the provided context from the video
    - If the context doesn't contain enough information to answer the question, say so
    - Be concise but comprehensive in your response
    - Use specific details from the video when possible
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create simple chain function
    def rag_chain_wrapper(inputs):
        question = inputs.get("question", "")
        
        # Get relevant documents
        docs = retriever.get_relevant_documents(question)
        context = _format_docs(docs)
        
        # Format prompt
        formatted_prompt = template.format(context=context, question=question)
        
        # Get response from LLM
        response = llm.invoke(formatted_prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {"answer": answer, "context": context}
    
    return rag_chain_wrapper

def _format_docs(docs):
    """Format retrieved documents for the prompt"""
    return "\n\n".join([doc.page_content for doc in docs])

def generate_suggested_questions_chain(temperature=0.7):
    """Create a chain to generate suggested questions."""
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama3-8b-8192",
        temperature=temperature
    )

    template = """Based on the following summary of a YouTube video transcript, generate exactly 4 relevant and insightful questions that a user might ask about the video's content. The questions should be directly answerable from the video's content.

    Present the questions as a Python list of strings. For example:
    ["What are the key arguments presented about [topic]?", "How does the speaker suggest to solve [problem]?", "Can you explain the process of [specific concept] mentioned in the video?", "What are the main takeaways regarding [subject]?"]

    Transcript summary:
    {transcript_summary}

    Suggested Questions:"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm | StrOutputParser()
    return chain

def create_summarization_chain(temperature=0.7):
    """Create a chain to summarize the transcript."""
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama3-8b-8192",
        temperature=temperature
    )

    template = """You are a helpful AI assistant that summarizes YouTube video transcripts.
    Based on the following transcript, please provide a concise summary of the key points.

    Transcript:
    {transcript}

    Summary:"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm | StrOutputParser()
    return chain