"""
Vision-based RAG System - Streamlit UI
Upload PDFs and ask questions using AI vision analysis
"""

import asyncio
import streamlit as st
from pathlib import Path
import tempfile
import os

from app.processor.pdf_vision import VisionPDFProcessor
from app.ai.vision_analysis import VisionAnalysisService
from app.chat.chat_service import ChatService
from app.providers.factory import create_provider_from_env


# Configure Streamlit page
st.set_page_config(
    page_title="Vision RAG - Document Q&A",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'chat_service' not in st.session_state:
    st.session_state.chat_service = None
if 'current_document' not in st.session_state:
    st.session_state.current_document = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing' not in st.session_state:
    st.session_state.processing = False


def initialize_services():
    """Initialize chat service if not already done"""
    if st.session_state.chat_service is None:
        with st.spinner("Initializing services..."):
            try:
                st.session_state.chat_service = ChatService()
                st.success("✅ Services initialized!")
            except Exception as e:
                st.error(f"❌ Failed to initialize: {e}")
                return False
    return True


async def process_uploaded_file(uploaded_file):
    """Process uploaded PDF file"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        # Initialize processor and vision service
        processor = VisionPDFProcessor()
        provider = create_provider_from_env()
        vision_service = VisionAnalysisService(provider)

        # Process PDF
        st.info("📄 Converting PDF pages to images...")
        document = await processor.process(tmp_path)
        
        # Analyze document
        st.info("🔍 Analyzing document with AI vision...")
        progress_bar = st.progress(0)
        
        # Analyze each page
        for i, page in enumerate(document.pages):
            context = f"This is page {page.page_number} of {document.page_count} from '{document.name}'"
            page.summary = await vision_service.analyze_page(page, context)
            progress_bar.progress((i + 1) / len(document.pages))
        
        # Save document
        document.summary = f"Document with {len(document.pages)} pages analyzed"
        vision_service._save_document_metadata(document)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return document

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        return None


def display_document_info(doc_info):
    """Display document information"""
    st.markdown(f"**📄 {doc_info['name']}**")
    st.markdown(f"- Pages: {doc_info['page_count']}")
    st.markdown(f"- Status: {doc_info['status']}")
    st.markdown(f"- Has Summaries: {'✅' if doc_info['has_summaries'] else '❌'}")


def main():
    """Main Streamlit app"""
    
    # Header
    st.title("📄 Vision RAG - Document Q&A System")
    st.markdown("Upload PDFs and ask questions using AI vision analysis")
    
    # Initialize services
    if not initialize_services():
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📚 Documents")
        
        # Upload section
        st.subheader("Upload New PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF to analyze and ask questions about"
        )
        
        if uploaded_file and st.button("🚀 Process Document", type="primary"):
            st.session_state.processing = True
            with st.spinner("Processing document..."):
                document = asyncio.run(process_uploaded_file(uploaded_file))
                if document:
                    st.success(f"✅ Document processed: {document.name}")
                    st.session_state.current_document = document.id
                    st.session_state.chat_history = []
                    st.rerun()
            st.session_state.processing = False
        
        # List existing documents
        st.subheader("Existing Documents")
        documents = st.session_state.chat_service.list_documents()
        
        if documents:
            for doc in documents:
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(
                        f"📄 {doc['name'][:30]}...",
                        key=f"doc_{doc['id']}",
                        help=f"{doc['page_count']} pages"
                    ):
                        st.session_state.current_document = doc['id']
                        st.session_state.chat_history = []
                        st.rerun()
                with col2:
                    if doc['has_summaries']:
                        st.markdown("✅")
                    else:
                        st.markdown("⚠️")
        else:
            st.info("No documents yet. Upload one above!")
        
        # Cost tracking
        st.divider()
        total_cost = st.session_state.chat_service.get_total_cost()
        st.metric("💰 Session Cost", f"${total_cost:.4f}")
        if st.button("🔄 Reset Cost"):
            st.session_state.chat_service.reset_cost()
            st.rerun()
    
    # Main content area
    if st.session_state.current_document:
        # Get document info
        doc_info = st.session_state.chat_service.get_document_info(
            st.session_state.current_document
        )
        
        if doc_info:
            # Agent always decides - no settings, no header
            max_pages = None
            use_all_pages = False
            
            # Display chat history
            chat_container = st.container()
            with chat_container:
                if st.session_state.chat_history:
                    for i, chat in enumerate(st.session_state.chat_history, 1):
                        # User message
                        with st.chat_message("user"):
                            st.markdown(chat['question'])
                        
                        # Assistant response
                        result = chat['result']
                        with st.chat_message("assistant"):
                            st.markdown(result['answer'])
                            
                            # Show metadata in a compact format
                            page_list = ", ".join(map(str, result['page_numbers']))
                            st.caption(
                                f"📄 Pages: {page_list if page_list else 'None'} | "
                                f"💰 Cost: ${result['total_cost']:.4f} | "
                                f"📊 {len(result['page_numbers'])} pages analyzed"
                            )
        
            
            # Chat input at the bottom
            st.divider()
            
            # Input area
            col1, col2 = st.columns([5, 1])
            with col1:
                question = st.text_input(
                    "Ask a question:",
                    placeholder="Type your question here...",
                    key="question_input",
                    label_visibility="collapsed"
                )
            with col2:
                send_button = st.button("📤 Send", type="primary", use_container_width=True)
            
            # Process question
            if send_button and question:
                with st.spinner("🤔 Thinking..."):
                    try:
                        result = asyncio.run(
                            st.session_state.chat_service.ask(
                                document_id=st.session_state.current_document,
                                question=question,
                                max_pages=max_pages,
                                use_all_pages=use_all_pages
                            )
                        )
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'question': question,
                            'result': result
                        })
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
            
            # Action buttons
            if st.session_state.chat_history:
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("🗑️ Clear Chat", use_container_width=True):
                        st.session_state.chat_history = []
                        st.rerun()
                with col2:
                    if st.button("📊 Stats", use_container_width=True):
                        total_questions = len(st.session_state.chat_history)
                        total_cost = sum(chat['result']['total_cost'] for chat in st.session_state.chat_history)
                        total_pages = sum(len(chat['result']['page_numbers']) for chat in st.session_state.chat_history)
                        avg_pages = total_pages / total_questions if total_questions > 0 else 0
                        
                        st.success(
                            f"**💬 {total_questions} questions** | "
                            f"**📄 {total_pages} pages** | "
                            f"**💰 ${total_cost:.4f}** | "
                            f"**📊 {avg_pages:.1f} avg**"
                        )
    
    else:
        # Welcome screen
        st.info("👈 Select or upload a document from the sidebar to get started!")
        
        st.markdown("""
        ### How to use:
        
        1. **Upload a PDF** using the sidebar
        2. **Wait for processing** - the system will:
           - Convert pages to images
           - Analyze each page with AI vision
           - Generate summaries
        3. **Ask questions** about the document
        4. **Get answers** based on relevant pages
        
        ### Features:
        
        - 🤖 **Smart Agent** - Automatically decides how many pages to analyze
        - 👁️ **Vision Analysis** - Understands tables, charts, and formatting
        - 💰 **Cost Tracking** - Monitor API usage in real-time
        - 💬 **Conversational UI** - Natural chat interface
        - 📊 **Session Statistics** - Track your analysis metrics
        """)


if __name__ == "__main__":
    main()