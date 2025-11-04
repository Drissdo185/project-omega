# ai/vision_analyzer.py
import base64
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from loguru import logger

from processors.document import (
    Document,
    Page,
    TableInfo,
    ChartInfo,
    Partition,
    PartitionDetails,
    PartitionDetail,
    TableInfoWithPage,
    ChartInfoWithPage,
)
from openai import OpenAIClient


class VisionAnalyzer:
    """AI-powered vision analysis for document pages"""

    def __init__(self, openai_client: OpenAIClient, storage_root: str = None):
        self.client = openai_client
        
        if storage_root is None:
            import os
            storage_root = os.environ.get("FLEX_RAG_DATA_LOCATION", "../../flex_rag_data_location")
        
        self.storage_root = Path(storage_root)
        self.documents_dir = self.storage_root / "documents"

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def _analyze_single_page(self, image_path: str, page_number: int) -> Dict:
        """Analyze a single page image"""
        try:
            base64_image = self._encode_image(image_path)
            
            prompt = f"""Analyze this document page (page {page_number}) and extract:

1. **Page Summary**: Write a comprehensive 2-3 sentence summary of the main content on this page.

2. **Tables**: Identify ALL tables on this page. For each table:
   - Assign table_id as "table_{page_number}_N" (N = 1, 2, 3...)
   - Provide a descriptive title
   - Write a summary explaining what data the table contains

3. **Charts/Graphs**: Identify ALL charts, graphs, or visualizations. For each:
   - Assign chart_id as "chart_{page_number}_N" (N = 1, 2, 3...)
   - Provide a descriptive title
   - Identify chart_type (one of: line, bar, pie, scatter, area, other)
   - Write a summary explaining what the chart shows

Return ONLY valid JSON in this exact format:
{{
  "summary": "Page summary here...",
  "tables": [
    {{
      "table_id": "table_{page_number}_1",
      "title": "Table title",
      "summary": "Table summary"
    }}
  ],
  "charts": [
    {{
      "chart_id": "chart_{page_number}_1",
      "title": "Chart title",
      "chart_type": "line",
      "summary": "Chart summary"
    }}
  ]
}}

If there are no tables or charts, use empty arrays []."""

            response = await self.client.vision_completion(
                text_prompt=prompt,
                images=[base64_image],
                model=self.client.model_small,
                max_tokens=1500,
                detail="high"
            )

            # Parse JSON
            content = response.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            
            logger.info(f"✅ Analyzed page {page_number}: {len(result.get('tables', []))} tables, {len(result.get('charts', []))} charts")
            
            return result

        except Exception as e:
            logger.error(f"Failed to analyze page {page_number}: {e}")
            return {"summary": "", "tables": [], "charts": []}

    async def _analyze_partition_batch(
        self, 
        pages: List[Page], 
        partition_id: int,
        page_range: Tuple[int, int]
    ) -> Dict:
        """
        Analyze a partition (batch of pages) for partition summary
        
        Args:
            pages: List of Page objects in this partition
            partition_id: Partition ID
            page_range: (start_page, end_page) tuple
            
        Returns:
            {
                "summary": "Partition summary (2-3 sentences)..."
            }
        """
        try:
            # Sample pages if too many (max 10 images for partition analysis)
            max_images = 10
            sample_pages = pages[:max_images] if len(pages) > max_images else pages
            
            # Encode sample images
            images = []
            for page in sample_pages:
                try:
                    base64_image = self._encode_image(page.image_path)
                    images.append(base64_image)
                except Exception as e:
                    logger.warning(f"Failed to encode page {page.page_number} for partition analysis: {e}")
            
            if not images:
                logger.error(f"No images available for partition {partition_id}")
                return {"summary": ""}
            
            # Build context from individual page summaries
            page_summaries_context = "\n".join([
                f"Page {p.page_number}: {p.summary}" for p in sample_pages if p.summary
            ])
            
            prompt = f"""Analyze this document partition (Partition {partition_id}: Pages {page_range[0]}-{page_range[1]}).

Total pages in partition: {len(pages)}
{"You are viewing all pages." if len(pages) <= max_images else f"You are viewing a sample of {len(sample_pages)} pages."}

Individual page summaries:
{page_summaries_context}

Based on the page images and summaries, provide a comprehensive 2-3 sentence summary that:
- Captures the main topics and themes covered in this partition
- Highlights key information, data, or findings presented
- Describes the overall purpose or focus of this section

Return ONLY valid JSON:
{{
  "summary": "2-3 sentence comprehensive summary of this partition..."
}}"""

            response = await self.client.vision_completion(
                text_prompt=prompt,
                images=images,
                model=self.client.model_large,
                max_tokens=500,
                detail="low",  # Use low detail for partition overview
                temperature=0.3
            )

            # Parse JSON
            content = response.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            
            logger.info(f"✅ Analyzed partition {partition_id} (pages {page_range[0]}-{page_range[1]})")
            
            return result

        except Exception as e:
            logger.error(f"Failed to analyze partition {partition_id}: {e}")
            return {"summary": ""}

    async def analyze_document(self, document: Document) -> Document:
        """
        Main method to analyze a document
        
        For ≤20 pages:
        - Analyze each page individually
        - Update metadata.json
        
        For >20 pages:
        - Analyze each page individually
        - Analyze each partition for summaries
        - Update metadata.json
        - Create partition_summary.json
        
        Args:
            document: Document object with pages
            
        Returns:
            Updated Document object
        """
        doc_dir = self.documents_dir / document.id
        
        logger.info("=" * 80)
        logger.info(f"🤖 Starting AI analysis for: {document.name}")
        logger.info(f"📊 Total pages: {document.page_count}")
        logger.info(f"📁 Document ID: {document.id}")
        logger.info("=" * 80)
        
        # Step 1: Analyze each page individually
        logger.info(f"\n📄 Step 1: Analyzing individual pages...")
        
        for idx, page in enumerate(document.pages, 1):
            logger.info(f"Processing page {idx}/{len(document.pages)} (Page {page.page_number})...")
            
            analysis = await self._analyze_single_page(page.image_path, page.page_number)
            
            # Update page with analysis results
            page.summary = analysis.get("summary", "")
            
            # Add tables
            page.tables = [
                TableInfo(
                    table_id=t["table_id"],
                    title=t["title"],
                    summary=t["summary"]
                )
                for t in analysis.get("tables", [])
            ]
            
            # Add charts
            page.charts = [
                ChartInfo(
                    chart_id=c["chart_id"],
                    title=c["title"],
                    chart_type=c["chart_type"],
                    summary=c["summary"]
                )
                for c in analysis.get("charts", [])
            ]
        
        logger.info(f"✅ Completed page-level analysis for all {len(document.pages)} pages")
        
        # Count total tables and charts
        total_tables = sum(len(p.tables) for p in document.pages)
        total_charts = sum(len(p.charts) for p in document.pages)
        logger.info(f"📋 Total tables detected: {total_tables}")
        logger.info(f"📈 Total charts detected: {total_charts}")
        
        # Step 2: For large documents, analyze partitions
        if document.has_partitions():
            logger.info(f"\n📑 Step 2: Analyzing {len(document.partitions)} partitions...")
            
            for partition in document.partitions:
                logger.info(f"Analyzing partition {partition.partition_id}/{len(document.partitions)}...")
                
                # Get pages for this partition
                partition_pages = [
                    p for p in document.pages
                    if p.partition_id == partition.partition_id
                ]
                
                logger.info(f"  Pages in partition: {len(partition_pages)}")
                
                # Analyze partition
                partition_analysis = await self._analyze_partition_batch(
                    partition_pages,
                    partition.partition_id,
                    partition.page_range
                )
                
                # Update partition summary
                partition.summary = partition_analysis.get("summary", "")
                
                if partition.summary:
                    logger.info(f"  ✅ Summary: {partition.summary[:100]}...")
            
            logger.info(f"✅ Completed partition-level analysis")
        
        # Step 3: Save updated metadata.json
        logger.info(f"\n💾 Step 3: Saving results...")
        self._save_metadata(document, doc_dir)
        
        # Step 4: For large documents, create partition_summary.json
        if document.has_partitions():
            self._save_partition_summary(document, doc_dir)
        
        logger.info("=" * 80)
        logger.info(f"✅ AI ANALYSIS COMPLETE")
        logger.info(f"📄 Document: {document.name}")
        logger.info(f"📊 Pages analyzed: {document.page_count}")
        logger.info(f"📋 Tables found: {total_tables}")
        logger.info(f"📈 Charts found: {total_charts}")
        if document.has_partitions():
            logger.info(f"📑 Partitions: {len(document.partitions)}")
        logger.info("=" * 80)
        
        return document

    def _save_metadata(self, document: Document, doc_dir: Path) -> None:
        """
        Save updated metadata.json
        
        Args:
            document: Document object
            doc_dir: Document directory path
        """
        try:
            metadata_path = doc_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(document.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved metadata to {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise

    def _save_partition_summary(self, document: Document, doc_dir: Path) -> None:
        """
        Create and save partition_summary.json for large documents
        
        Args:
            document: Document object
            doc_dir: Document directory path
        """
        try:
            partition_details_list = []
            
            for partition in document.partitions:
                # Get all pages in this partition
                partition_pages = [
                    p for p in document.pages
                    if p.partition_id == partition.partition_id
                ]
                
                # Aggregate tables with page numbers
                tables = []
                for page in partition_pages:
                    for table in page.tables:
                        tables.append(TableInfoWithPage(
                            table_id=table.table_id,
                            page_number=page.page_number,
                            title=table.title,
                            summary=table.summary
                        ))
                
                # Aggregate charts with page numbers
                charts = []
                for page in partition_pages:
                    for chart in page.charts:
                        charts.append(ChartInfoWithPage(
                            chart_id=chart.chart_id,
                            page_number=page.page_number,
                            title=chart.title,
                            chart_type=chart.chart_type,
                            summary=chart.summary
                        ))
                
                # Create PartitionDetail
                partition_detail = PartitionDetail(
                    partition_id=partition.partition_id,
                    page_range=partition.page_range,
                    page_count=partition.get_page_count(),
                    summary=partition.summary,
                    tables=tables,
                    charts=charts
                )
                partition_details_list.append(partition_detail)
                
                logger.info(f"  Partition {partition.partition_id}: {len(tables)} tables, {len(charts)} charts")
            
            # Create PartitionDetails container
            partition_summary = PartitionDetails(
                document_id=document.id,
                document_name=document.name,
                total_partitions=len(document.partitions),
                partitions=partition_details_list
            )
            
            # Save to partition_summary.json
            summary_path = doc_dir / "partition_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(partition_summary.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved partition_summary.json to {summary_path}")
            logger.info(f"📊 Total partitions in summary: {len(partition_details_list)}")
        
        except Exception as e:
            logger.error(f"Failed to save partition summary: {e}")
            raise