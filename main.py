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
from app.utils.document_manager import reanalyze_document


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
if 'reanalyzing' not in st.session_state:
    st.session_state.reanalyzing = False


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
        st.info("🔍 Analyzing document with AI vision (gpt-oss-20b)...")
        progress_bar = st.progress(0)
        
        # Group pages by combined image for analysis
        image_to_pages = {}
        for page in document.pages:
            image_path = page.image_path
            if image_path not in image_to_pages:
                image_to_pages[image_path] = []
            image_to_pages[image_path].append(page)
        
        # Analyze each combined image
        analyzed_images = 0
        total_images = len(image_to_pages)
        
        for image_path, pages_in_image in image_to_pages.items():
            context = f"Combined image from '{document.name}' containing pages {[p.page_number for p in pages_in_image]}"
            
            # Analyze all pages in this combined image
            page_analyses = await vision_service.analyze_combined_image(
                image_path,
                pages_in_image,
                context
            )
            
            # Update page objects with analysis results
            for analysis in page_analyses:
                for page in pages_in_image:
                    if page.page_number == analysis["page_number"]:
                        page.summary = analysis["summary"]
                        page.isImage = analysis["isImage"]
                        # Update dimensions if provided
                        if analysis.get("width"):
                            page.width = analysis["width"]
                        if analysis.get("height"):
                            page.height = analysis["height"]
                        break
            
            analyzed_images += 1
            progress_bar.progress(analyzed_images / total_images)
        
        # Save document with updated metadata
        pages_with_images = len([p for p in document.pages if p.isImage])
        document.summary = f"Document with {len(document.pages)} pages analyzed using gpt-oss-20b, {pages_with_images} pages contain images"
        vision_service._save_document_metadata(document)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return document

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        return None


def display_document_info(doc_info):
    """Display document information (optimized)"""
    st.markdown(f"**📄 {doc_info['name']}**")
    st.markdown(f"- Pages: {doc_info['page_count']}")
    st.markdown(f"- Status: {doc_info['status']}")
    
    # Use pre-computed has_summaries if available
    if 'has_summaries' in doc_info:
        has_summaries = doc_info['has_summaries']
    else:
        # Fallback to checking pages
        has_summaries = any(page.get('summary') for page in doc_info.get('pages', []))
    
    st.markdown(f"- Has Summaries: {'✅' if has_summaries else '❌'}")
    if 'pages_with_images' in doc_info:
        st.markdown(f"- Pages with Images: {doc_info['pages_with_images']}")
    if 'combined_images_count' in doc_info:
        st.markdown(f"- Combined Images: {doc_info['combined_images_count']}")


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
        
        # List existing documents (cached)
        st.subheader("Existing Documents")
        
        # Cache document list in session state for better performance
        if 'documents_list' not in st.session_state or st.session_state.processing or st.session_state.reanalyzing:
            documents = st.session_state.chat_service.list_documents()
            st.session_state.documents_list = documents
        else:
            documents = st.session_state.documents_list
        
        if documents:
            # Add document selector
            doc_names = [f"{doc['name']} ({doc['page_count']} pages)" for doc in documents]
            doc_ids = [doc['id'] for doc in documents]
            
            # Find current selection index
            current_index = 0
            if st.session_state.current_document:
                try:
                    current_index = doc_ids.index(st.session_state.current_document)
                except ValueError:
                    current_index = 0
            
            # Document selector
            selected_index = st.selectbox(
                "📋 Select document to ask questions:",
                range(len(doc_names)),
                format_func=lambda x: doc_names[x],
                index=current_index,
                help="Choose which document you want to ask questions about"
            )
            
            # Update current document if selection changed
            if doc_ids[selected_index] != st.session_state.current_document:
                st.session_state.current_document = doc_ids[selected_index]
                st.session_state.chat_history = []
                st.rerun()
            
            # Show reanalysis status if in progress
            if st.session_state.reanalyzing:
                st.warning("⏳ Reanalysis in progress... Please wait.")
            
            # Show detailed info for selected document
            selected_doc = documents[selected_index]
            with st.expander("📊 Document Details", expanded=False):
                st.markdown(f"**Name:** {selected_doc['name']}")
                st.markdown(f"**Pages:** {selected_doc['page_count']}")
                st.markdown(f"**Status:** {selected_doc['status']}")
                has_summaries = selected_doc.get('has_summaries', False)
                st.markdown(f"**Has Analysis:** {'✅ Ready' if has_summaries else '❌ Pending'}")
                if 'created_at' in selected_doc:
                    st.markdown(f"**Created:** {selected_doc['created_at']}")
                
                # Quick action buttons
                col1, col2 = st.columns(2)
                with col1:
                    reanalyze_disabled = st.session_state.processing or st.session_state.reanalyzing
                    if st.button("🔄 Reanalyze", 
                                key=f"reanalyze_{selected_doc['id']}", 
                                help="Re-process this document with updated analysis",
                                disabled=reanalyze_disabled,
                                use_container_width=True):
                        st.session_state.reanalyzing = True
                        success = asyncio.run(reanalyze_document(selected_doc['id']))
                        st.session_state.reanalyzing = False
                        
                        # Clear chat history if reanalysis was successful
                        if success and st.session_state.current_document == selected_doc['id']:
                            st.session_state.chat_history = []
                        
                        st.rerun()
                with col2:
                    if st.button("🗑️ Delete", 
                                key=f"delete_{selected_doc['id']}", 
                                help="Remove this document from the system",
                                use_container_width=True):
                        st.info("🗑️ Delete feature - implement if needed")
            
            # Alternative: Grid view for documents (comment out if not needed)
            st.markdown("---")
            st.markdown("**📂 Quick Access:**")
            cols = st.columns(min(3, len(documents)))
            for i, doc in enumerate(documents):
                with cols[i % 3]:
                    # Determine status emoji
                    status_emoji = "✅" if doc['has_summaries'] else "⚠️"
                    
                    # Current document indicator
                    border_style = "border: 2px solid #00ff00; border-radius: 5px; padding: 8px;" if doc['id'] == st.session_state.current_document else "border: 1px solid #ccc; border-radius: 5px; padding: 8px;"
                    
                    # Create document card
                    st.markdown(f"""
                    <div style="{border_style}">
                        <div style="font-size: 14px; font-weight: bold;">{doc['name'][:20]}{'...' if len(doc['name']) > 20 else ''}</div>
                        <div style="font-size: 12px; color: #666;">{doc['page_count']} pages</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Select", key=f"quick_select_{doc['id']}", use_container_width=True):
                        st.session_state.current_document = doc['id']
                        st.session_state.chat_history = []
                        st.rerun()
        else:
            st.info("No documents yet. Upload one above!")
        
        # Current document status
        if st.session_state.current_document:
            st.divider()
            st.markdown("**🎯 Currently Active:**")
            current_doc = next((doc for doc in documents if doc['id'] == st.session_state.current_document), None)
            if current_doc:
                st.success(f"📄 {current_doc['name'][:25]}{'...' if len(current_doc['name']) > 25 else ''}")
                st.caption(f"Ready for questions • {current_doc['page_count']} pages")
            else:
                st.warning("⚠️ Selected document not found")
                st.session_state.current_document = None
        
        # Cost tracking
        st.divider()
        total_cost = st.session_state.chat_service.get_total_cost()
        st.metric("💰 Session Cost", f"${total_cost:.4f}")
        if st.button("🔄 Reset Cost"):
            st.session_state.chat_service.reset_cost()
            st.rerun()
    
    # Main content area
    if st.session_state.current_document:
        # Get document info (cached in session state)
        cache_doc_key = f'doc_info_{st.session_state.current_document}'
        
        if cache_doc_key not in st.session_state or st.session_state.processing or st.session_state.reanalyzing:
            doc_info = st.session_state.chat_service.get_document_info(
                st.session_state.current_document
            )
            st.session_state[cache_doc_key] = doc_info
        else:
            doc_info = st.session_state[cache_doc_key]
        
        if doc_info:
            # Document header with current selection
            st.header(f"💬 Chat with: {doc_info['name']}")
            
            # Document stats in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📄 Pages", doc_info['page_count'])
            with col2:
                # Check if document has summaries by examining pages
                has_summaries = any(page.get('summary') for page in doc_info.get('pages', []))
                st.metric("🔍 Status", "✅ Ready" if has_summaries else "⚠️ Processing")
            with col3:
                pages_with_images = doc_info.get('pages_with_images', 0)
                st.metric("🖼️ Images", pages_with_images)
            with col4:
                if st.session_state.chat_history:
                    total_q = len(st.session_state.chat_history)
                    st.metric("💬 Questions", total_q)
                else:
                    st.metric("💬 Questions", 0)
            
            st.divider()
            
            # Agent always decides - no settings, no header
            max_pages = None
            use_all_pages = False
            
            # Display chat history
            if st.session_state.chat_history:
                st.markdown("### 💬 Conversation")
                for i, chat in enumerate(st.session_state.chat_history, 1):
                    # User message
                    with st.chat_message("user"):
                        st.write(chat['question'])
                    
                    # Assistant response
                    result = chat['result']
                    with st.chat_message("assistant"):
                        # Display answer with validation
                        answer = result.get('answer', '')
                        
                        if not answer or not answer.strip():
                            st.error("⚠️ The AI returned an empty response. This could mean:")
                            st.markdown("""
                            - The API endpoint may not support vision/multimodal requests
                            - The selected pages may not contain relevant information
                            - There might be an API configuration issue
                            
                            **Try:**
                            - Rephrasing your question
                            - Checking if the document was analyzed properly
                            - Verifying your API endpoint supports multimodal requests
                            """)
                        else:
                            st.write(answer)
                        
                        # Show metadata in a compact format
                        page_numbers = result.get('page_numbers', [])
                        page_list = ", ".join(map(str, page_numbers))
                        total_cost = result.get('total_cost', 0)
                        
                        st.caption(
                            f"📄 Pages: {page_list if page_list else 'None'} | "
                            f"💰 Cost: ${total_cost:.4f} | "
                            f"📊 {len(page_numbers)} pages analyzed"
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
        
        # Show existing documents if any
        documents = st.session_state.chat_service.list_documents()
        if documents:
            st.markdown("### 📚 Available Documents:")
            
            # Show documents in a grid
            cols = st.columns(min(3, len(documents)))
            for i, doc in enumerate(documents):
                with cols[i % 3]:
                    status_emoji = "✅" if doc['has_summaries'] else "⚠️"
                    
                    with st.container():
                        st.markdown(f"""
                        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 5px; text-align: center;">
                            <div style="font-size: 18px;">{status_emoji}</div>
                            <div style="font-weight: bold; margin: 5px 0;">{doc['name'][:25]}{'...' if len(doc['name']) > 25 else ''}</div>
                            <div style="color: #666; font-size: 12px;">{doc['page_count']} pages</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button(f"📖 Open {doc['name'][:15]}{'...' if len(doc['name']) > 15 else ''}", 
                                   key=f"welcome_select_{doc['id']}", 
                                   use_container_width=True):
                            st.session_state.current_document = doc['id']
                            st.session_state.chat_history = []
                            st.rerun()
            
            st.markdown("---")
        
        st.markdown("""
        ### How to use:
        
        1. **📤 Upload a PDF** using the sidebar upload section
        2. **⏳ Wait for processing** - the system will:
           - Convert pages to images (20 pages per combined image)
           - Analyze each page with AI vision (GPT-4o-mini)
           - Generate detailed summaries
        3. **📋 Select a document** from the dropdown or quick access buttons
        4. **❓ Ask questions** about the selected document content
        5. **💬 Get answers** based on relevant pages
        
        ### ✨ Features:
        
        - 🤖 **Smart Document Selection** - Choose which document to query
        - 👁️ **Vision Analysis** - Understands tables, charts, and text formatting
        - 🎯 **Intelligent Page Selection** - AI automatically finds relevant pages
        - 💰 **Cost Tracking** - Monitor API usage in real-time
        - 💬 **Conversational Interface** - Natural chat experience
        - 📊 **Session Statistics** - Track analysis metrics per document
        - 🔄 **Document Management** - Easy switching between multiple documents
        """)
        
        # Quick tips
        with st.expander("💡 Pro Tips", expanded=False):
            st.markdown("""
            - **Multiple Documents**: Upload multiple PDFs and switch between them easily
            - **Document Status**: ✅ = Ready for questions, ⚠️ = Still processing  
            - **Question Types**: Ask about specific topics, request summaries, or find detailed information
            - **Page References**: The AI will show which pages were used to answer your question
            - **Cost Management**: Monitor your API usage with the cost tracker in the sidebar
            """)


if __name__ == "__main__":
    main()