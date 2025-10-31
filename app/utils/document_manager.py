"""
Document Management Utilities
"""

import streamlit as st
from app.ai.vision_analysis import VisionAnalysisService
from app.providers.factory import create_provider_from_env


async def reanalyze_document(document_id: str) -> bool:
    """
    Re-analyze an existing document with updated AI analysis
    
    Args:
        document_id: ID of the document to reanalyze
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize services
        provider = create_provider_from_env()
        vision_service = VisionAnalysisService(provider)
        
        # Load the existing document
        document = vision_service.load_document(document_id)
        
        if not document:
            st.error(f"❌ Document not found: {document_id}")
            return False
        
        st.info(f"🔄 Re-analyzing: {document.name}")
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Group pages by combined image
        image_to_pages = {}
        for page in document.pages:
            image_path = page.image_path
            if image_path not in image_to_pages:
                image_to_pages[image_path] = []
            image_to_pages[image_path].append(page)
        
        # Re-analyze each combined image
        analyzed_images = 0
        total_images = len(image_to_pages)
        
        for image_path, pages_in_image in image_to_pages.items():
            status_text.text(f"🔍 Analyzing image {analyzed_images + 1}/{total_images} ({len(pages_in_image)} pages)...")
            
            context = f"Re-analyzing '{document.name}' pages {[p.page_number for p in pages_in_image]}"
            
            # Analyze pages in this combined image
            page_analyses = await vision_service.analyze_combined_image(
                image_path,
                pages_in_image,
                context
            )
            
            # Update page objects with new analysis
            for analysis in page_analyses:
                for page in pages_in_image:
                    if page.page_number == analysis["page_number"]:
                        page.summary = analysis["summary"]
                        page.isImage = analysis["isImage"]
                        if analysis.get("width"):
                            page.width = analysis["width"]
                        if analysis.get("height"):
                            page.height = analysis["height"]
                        break
            
            analyzed_images += 1
            progress_bar.progress(analyzed_images / total_images)
        
        # Update document summary
        pages_with_images = len([p for p in document.pages if p.isImage])
        document.summary = f"Document with {len(document.pages)} pages analyzed, {pages_with_images} pages contain images/graphics"
        
        # Save updated metadata
        vision_service._save_document_metadata(document)
        
        # Clear progress indicators
        status_text.empty()
        progress_bar.empty()
        
        st.success(f"✅ Re-analyzed: {document.name}")
        st.info(f"📊 {document.page_count} pages, {pages_with_images} with images")
        
        # Show cost
        cost = vision_service.provider.get_last_cost()
        if cost:
            st.info(f"💰 Cost: ${cost:.4f}")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Re-analysis failed: {str(e)}")
        return False


