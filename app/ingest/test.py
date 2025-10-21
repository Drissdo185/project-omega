# quick_test.py
"""Quick test for PDF vision processor"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from app.ingest.factory import ProcessorFactory

async def main():
    # Get PDF path
    pdf_path = "/home/dtdat/Desktop/Project/project-omega/data/TrinhBinhNguyen_CV.pdf"
    
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        print("Usage: python quick_test.py <pdf_file>")
        return
    
    print(f"📄 Processing: {pdf_path}")
    
    # Create processor and process PDF
    processor = ProcessorFactory.create_processor(pdf_path)
    pages = await processor.process(pdf_path)
    
    # Show results
    print(f"✅ Processed {len(pages)} pages:\n")
    for page in pages:
        size_kb = Path(page.image_path).stat().st_size / 1024
        print(f"  Page {page.page_number}: {page.width}x{page.height}px, {size_kb:.1f}KB")
        print(f"    → {page.image_path}")
    
    print(f"\n🎉 Done!")

if __name__ == "__main__":
    asyncio.run(main())