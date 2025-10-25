import asyncio

from app.processor.pdf_vision import VisionPDFProcessor
from app.ai.vision_analysis import VisionAnalysisService
from app.providers.factory import create_provider_from_env


async def main():

    processor = VisionPDFProcessor()
    provider = create_provider_from_env()
    vision_service = VisionAnalysisService(provider)

    document = await processor.process("/home/dtdat/Desktop/Project/project-omega/data/TrinhBinhNguyen_CV.pdf")

    analysis = await vision_service.analyze_document(document)

    # Results automatically saved to metadata.json
    print(f"Document: {analysis.document_name}")
    print(f"Summary: {analysis.overall_summary}")
    print(f"Total Cost: ${analysis.total_cost:.4f}")
    print(f"Page Analyses: {len(analysis.page_analyses)} pages")

asyncio.run(main())