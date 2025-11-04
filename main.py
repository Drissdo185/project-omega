# main.py
import streamlit as st
import asyncio
import os
import json
from pathlib import Path
from datetime import datetime
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the app directory to Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

from loguru import logger
from app.processors.pdf_to_image import VisionPDFProcessor
from app.ai.vision_analyzer import VisionAnalyzer
from app.ai.openai import OpenAIClient
from app.ai.page_selection_agent import PageSelectionAgent
from app.processors.document import Document

# Configure logger
logger.add("app.log", rotation="10 MB")

# Page config
st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if "document" not in st.session_state:
    st.session_state.document = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processing" not in st.session_state:
    st.session_state.processing = False


async def process_pdf(pdf_file, openai_api_key: str) -> Document:
    """Process uploaded PDF file"""
    
    # Save uploaded file temporarily
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    temp_path = temp_dir / pdf_file.name
    with open(temp_path, "wb") as f:
        f.write(pdf_file.getbuffer())
    
    try:
        # Step 1: Convert PDF to images
        with st.status("🔄 Converting PDF to images...", expanded=True) as status:
            st.write("Processing PDF pages...")
            
            processor = VisionPDFProcessor(
                render_scale=1.5,
                jpeg_quality=75,
                max_image_size=(1024, 1024),
                partition_size=20,
                partition_overlap=0
            )
            
            document = await processor.process(str(temp_path))
            
            st.write(f"✅ Converted {document.page_count} pages")
            if document.has_partitions():
                st.write(f"📑 Created {len(document.partitions)} partitions")
            
            status.update(label="✅ PDF conversion complete", state="complete")
        
        # Step 2: AI Analysis
        with st.status("🤖 Analyzing document with AI...", expanded=True) as status:
            st.write("Analyzing pages with vision AI...")
            
            openai_client = OpenAIClient(api_key=openai_api_key)
            analyzer = VisionAnalyzer(openai_client)
            
            document = await analyzer.analyze_document(document)
            
            # Count results
            total_tables = sum(len(p.tables) for p in document.pages)
            total_charts = sum(len(p.charts) for p in document.pages)
            
            st.write(f"✅ Analysis complete")
            st.write(f"📋 Found {total_tables} tables")
            st.write(f"📈 Found {total_charts} charts")
            
            status.update(label="✅ AI analysis complete", state="complete")
        
        return document
    
    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


async def answer_question(question: str, document: Document, openai_client: OpenAIClient):
    """Answer user question about the document"""
    
    agent = PageSelectionAgent(openai_client)
    
    # Select relevant pages
    with st.spinner("🔍 Finding relevant pages..."):
        selected_pages = await agent.select_relevant_pages(document, question)
    
    # Answer question
    with st.spinner("💭 Generating answer..."):
        result = await agent.answer_question(document, question, selected_pages)
    
    return result


def display_document_info(document: Document):
    """Display document information"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Pages", document.page_count)
    
    with col2:
        total_tables = sum(len(p.tables) for p in document.pages)
        st.metric("📋 Tables", total_tables)
    
    with col3:
        total_charts = sum(len(p.charts) for p in document.pages)
        st.metric("📈 Charts", total_charts)
    
    with col4:
        if document.has_partitions():
            st.metric("📑 Partitions", len(document.partitions))
        else:
            st.metric("📑 Partitions", "N/A")
    
    # Document details in expander
    with st.expander("📊 Document Details"):
        st.json({
            "ID": document.id,
            "Name": document.name,
            "Status": document.status.value,
            "Created": document.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "Has Partitions": document.has_partitions()
        })


def display_chat_message(role: str, content: str, metadata: dict = None):
    """Display a chat message"""
    
    with st.chat_message(role):
        st.markdown(content)
        
        if metadata:
            with st.expander("📎 Details"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Pages Used:**")
                    st.write(", ".join(map(str, metadata.get("pages_used", []))))
                
                with col2:
                    st.write("**Confidence:**")
                    confidence = metadata.get("confidence", "unknown")
                    emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
                    st.write(f"{emoji} {confidence.capitalize()}")


def main():
    """Main Streamlit app"""
    
    st.title("📄 PDF AI Assistant")
    st.markdown("Upload a PDF, process it with AI, and ask questions about its content.")
    
    # Get API key from environment
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")

    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        if openai_api_key:
            st.success("✅ OpenAI API Key loaded")
        else:
            st.error("❌ OPENAI_API_KEY not found")
            st.info("Set OPENAI_API_KEY in .env file")

        st.divider()

        # Current document info
        if st.session_state.document:
            st.header("📄 Current Document")
            st.success(f"✅ {st.session_state.document.name}")

            if st.button("🗑️ Clear Document"):
                st.session_state.document = None
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.info("No document loaded")
    
    # Main content area
    if not st.session_state.document:
        # Upload section
        st.header("1️⃣ Upload PDF Document")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a PDF document to analyze"
        )

        st.divider()

        # Show process button section
        if uploaded_file:
            st.subheader("2️⃣ Process Document")

            if not openai_api_key:
                st.error("❌ OPENAI_API_KEY not found in environment")
                st.info("Please set your OPENAI_API_KEY environment variable before running the app")
            else:
                st.info(f"📄 Ready to process: **{uploaded_file.name}**")

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button(
                        "🚀 Process Document with AI",
                        type="primary",
                        disabled=st.session_state.processing or not openai_api_key,
                        use_container_width=True
                    ):
                        st.session_state.processing = True

                        try:
                            # Process PDF
                            document = asyncio.run(process_pdf(uploaded_file, openai_api_key))
                            st.session_state.document = document
                            st.session_state.processing = False

                            st.success("✅ Document processed successfully!")
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Error processing document: {str(e)}")
                            logger.error(f"Processing error: {e}")
                            st.session_state.processing = False

                if st.session_state.processing:
                    st.warning("⏳ Processing in progress... Please wait.")
        else:
            st.info("👆 Please upload a PDF file to get started")
    
    else:
        # Document loaded - Show Q&A interface
        document = st.session_state.document
        
        # Display document info
        st.header("📊 Document Overview")
        display_document_info(document)
        
        st.divider()
        
        # Q&A Section
        st.header("💬 Ask Questions")
        
        # Display chat history
        for message in st.session_state.chat_history:
            display_chat_message(
                message["role"],
                message["content"],
                message.get("metadata")
            )
        
        # Chat input
        if question := st.chat_input("Ask a question about the document..."):
            # Add user message to chat
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })
            
            # Display user message
            display_chat_message("user", question)
            
            # Get answer
            try:
                openai_client = OpenAIClient(api_key=openai_api_key)
                result = asyncio.run(answer_question(question, document, openai_client))
                
                # Add assistant message to chat
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "metadata": {
                        "pages_used": result["pages_used"],
                        "confidence": result["confidence"]
                    }
                })
                
                # Display assistant message
                display_chat_message(
                    "assistant",
                    result["answer"],
                    {
                        "pages_used": result["pages_used"],
                        "confidence": result["confidence"]
                    }
                )
            
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                logger.error(f"Q&A error: {e}")
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg
                })


if __name__ == "__main__":
    main()