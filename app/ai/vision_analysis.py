"""
Vision-based document analysis service
Analyzes document images and generates summaries with tables/charts detection
"""
import json
from typing import List, Optional
from pathlib import Path
from loguru import logger
from app.models.document import Document, Page, TableInfo, ChartInfo, Partition, PartitionDetails, PartitionDetail, TableInfoWithPage, ChartInfoWithPage
from app.providers.base import BaseProvider


class VisionAnalysisService:
    """Service for analyzing documents using vision models"""

    def __init__(self, provider: BaseProvider, storage_root: str = None):
        """
        Initialize the vision analysis service
        
        Args:
            provider: LLM provider with vision capabilities
            storage_root: Root directory for storing analysis results
        """
        self.provider = provider

        if storage_root is None:
            import os
            storage_root = os.environ.get("FLEX_RAG_DATA_LOCATION", "./app/flex_rag_data_location")

        self.storage_root = Path(storage_root)
        self.documents_dir = self.storage_root / "documents"
        self.documents_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"VisionAnalysisService initialized with storage: {self.documents_dir}")

    async def analyze_page(
        self,
        page: Page,
        context: str = "",
        model: Optional[str] = None
    ) -> dict:
        """
        Analyze a single page using vision model and return structured data

        Args:
            page: Page object with image path
            context: Optional context about the document
            model: Optional model to use (if None, uses default from provider)

        Returns:
            dict: Contains summary, tables, and charts
        """
        try:
            logger.info(f"Analyzing page {page.page_number}: {page.image_path}")

            # Build the prompt
            prompt = self._build_analysis_prompt(context)

            # Prepare multimodal message
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert document analyst. Analyze images and provide structured information in JSON format."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_path", "image_path": page.image_path, "detail": "high"},
                    ],
                },
            ]

            # Get analysis from LLM
            response = await self.provider.process_multimodal_messages(
                messages=messages,
                max_tokens=1000,
                model=model
            )

            # Parse the response to get structured data
            page_data = self._parse_page_response(response, page.page_number)

            # Get cost
            cost = self.provider.get_last_cost() or 0.0
            logger.info(f"Page {page.page_number} analyzed (cost: ${cost:.4f})")

            return page_data

        except Exception as e:
            logger.error(f"Failed to analyze page {page.page_number}: {e}")
            return {
                "summary": "Analysis failed",
                "tables": [],
                "charts": []
            }

    async def analyze_document(
        self,
        document: Document,
        max_pages: Optional[int] = None,
        max_concurrent: int = 10  # Add concurrency control
    ) -> Document:
        """
        Analyze entire document with concurrent page processing and save metadata

        Args:
            document: Document object with pages
            max_pages: Optional limit on number of pages to analyze
            max_concurrent: Number of pages to process concurrently (default: 5)

        Returns:
            Document object with updated summaries, tables, and charts
        """
        try:
            logger.info(f"Starting analysis of document: {document.name} ({document.page_count} pages)")

            # Determine model based on document size
            # Large documents (>20 pages) use 3-stage model, small documents use 2-stage model
            model_for_analysis = self.provider.get_model_3stage() if document.is_large_document() else self.provider.get_model_2stage()
            logger.info(f"Using model {model_for_analysis} for document analysis (is_large: {document.is_large_document()})")

            pages_to_analyze = document.pages[:max_pages] if max_pages else document.pages

            # Process pages concurrently in batches
            import asyncio

            async def analyze_single_page(page: Page):
                """Analyze a single page and update it with results"""
                context = f"This is page {page.page_number} of {document.page_count} from '{document.name}'"
                page_data = await self.analyze_page(page, context, model=model_for_analysis)
                
                # Update page with results
                page.summary = page_data.get("summary", "")
                
                # Add tables
                for table_dict in page_data.get("tables", []):
                    table = TableInfo(
                        table_id=table_dict.get("table_id", f"table_{page.page_number}_1"),
                        title=table_dict.get("title", "Untitled Table"),
                        summary=table_dict.get("summary", "")
                    )
                    page.tables.append(table)
                
                # Add charts
                for chart_dict in page_data.get("charts", []):
                    chart = ChartInfo(
                        chart_id=chart_dict.get("chart_id", f"chart_{page.page_number}_1"),
                        title=chart_dict.get("title", "Untitled Chart"),
                        chart_type=chart_dict.get("chart_type", "unknown"),
                        summary=chart_dict.get("summary", "")
                    )
                    page.charts.append(chart)
            
            # Process in batches to avoid overwhelming the API
            total_batches = (len(pages_to_analyze) + max_concurrent - 1) // max_concurrent
            
            for i in range(0, len(pages_to_analyze), max_concurrent):
                batch = pages_to_analyze[i:i + max_concurrent]
                batch_num = i // max_concurrent + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} pages)")
                
                # Process all pages in the batch concurrently
                await asyncio.gather(*[analyze_single_page(page) for page in batch])
                
                logger.info(f"Completed batch {batch_num}/{total_batches}")
            
            # Create document-level summary
            total_tables = sum(len(p.tables) for p in pages_to_analyze)
            total_charts = sum(len(p.charts) for p in pages_to_analyze)
            document.summary = f"Document with {len(pages_to_analyze)} pages analyzed. Contains {total_tables} tables and {total_charts} charts."

            # Create partitions for large documents (>20 pages)
            if document.is_large_document():
                logger.info(f"Creating partitions for large document ({document.page_count} pages)")
                await self._create_partitions(document)

                # Create and save partition_details.json
                partition_details = self._create_partition_details(document)
                self._save_partition_details(partition_details, document.id)

            # Save complete metadata to JSON
            self._save_document_metadata(document)

            logger.info(f"Document analysis complete for: {document.name}")
            logger.info(f"Total: {len(pages_to_analyze)} pages, {total_tables} tables, {total_charts} charts")

            return document

        except Exception as e:
            logger.error(f"Failed to analyze document {document.id}: {e}")
            raise

    async def _create_partitions(self, document: Document, num_partitions: int = 5) -> None:
        """
        Create partition summaries for large documents.
        Divides document into equal partitions and generates summary for each.
        Also assigns partition_id to each page.

        Args:
            document: Document with analyzed pages
            num_partitions: Number of partitions to create (default: 5)
        """
        try:
            logger.info(f"Creating {num_partitions} partitions for {document.page_count} pages")

            # Calculate partition boundaries
            pages_per_partition = document.page_count / num_partitions
            partitions = []

            for i in range(num_partitions):
                start_page = int(i * pages_per_partition) + 1
                end_page = int((i + 1) * pages_per_partition) if i < num_partitions - 1 else document.page_count
                partition_id = i + 1

                # Get pages in this partition and assign partition_id
                partition_pages = []
                for page in document.pages:
                    if start_page <= page.page_number <= end_page:
                        page.partition_id = partition_id  # Assign partition_id to page
                        partition_pages.append(page)

                # Create partition summary from page summaries
                partition_summary = await self._summarize_partition(
                    partition_id=partition_id,
                    pages=partition_pages,
                    document_name=document.name
                )

                partition = Partition(
                    partition_id=partition_id,
                    page_range=(start_page, end_page),
                    summary=partition_summary
                )
                partitions.append(partition)

                logger.info(f"Partition {partition_id}: Pages {start_page}-{end_page} ({len(partition_pages)} pages)")

            document.partitions = partitions
            logger.info(f"Created {len(partitions)} partitions successfully")
            logger.info(f"Assigned partition_id to all {document.page_count} pages")

        except Exception as e:
            logger.error(f"Failed to create partitions: {e}")
            # Don't fail the entire analysis if partition creation fails
            document.partitions = []

    async def _summarize_partition(
        self,
        partition_id: int,
        pages: List[Page],
        document_name: str
    ) -> str:
        """
        Generate a summary for a partition based on its page summaries.
        Uses text-only LLM (cheap) to synthesize page summaries.

        Args:
            partition_id: Partition number
            pages: List of pages in this partition
            document_name: Name of the document

        Returns:
            Summary text for the partition
        """
        try:
            # Build context from page summaries
            page_summaries = []
            for page in pages:
                summary = page.summary or "No summary available"
                page_summaries.append(f"Page {page.page_number}: {summary}")

            pages_context = "\n".join(page_summaries)

            prompt = f"""You are analyzing partition {partition_id} of the document "{document_name}".

This partition contains pages {pages[0].page_number} to {pages[-1].page_number}.

Page summaries:
{pages_context}

Task: Create a concise summary (2-3 sentences) of what this partition covers.
Focus on the main topics, themes, or sections discussed in these pages.

Partition summary:"""

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert document analyst. Create concise partition summaries from page summaries."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            summary = await self.provider.process_text_messages(
                messages=messages,
                max_tokens=200
            )

            cost = self.provider.get_last_cost() or 0.0
            logger.debug(f"Partition {partition_id} summarized (cost: ${cost:.4f})")

            return summary.strip()

        except Exception as e:
            logger.error(f"Failed to summarize partition {partition_id}: {e}")
            return f"Partition {partition_id} covering pages {pages[0].page_number}-{pages[-1].page_number}"

    def _create_partition_details(self, document: Document) -> PartitionDetails:
        """
        Create PartitionDetails object by aggregating tables and charts from partition pages.

        Args:
            document: Document with partitions and analyzed pages

        Returns:
            PartitionDetails object for partition_details.json
        """
        partition_detail_list = []

        for partition in document.partitions:
            start_page, end_page = partition.page_range

            # Get all pages in this partition
            partition_pages = [p for p in document.pages if start_page <= p.page_number <= end_page]

            # Aggregate all tables from partition pages
            tables_with_page = []
            for page in partition_pages:
                for table in page.tables:
                    tables_with_page.append(TableInfoWithPage(
                        table_id=table.table_id,
                        page_number=page.page_number,
                        title=table.title,
                        summary=table.summary
                    ))

            # Aggregate all charts from partition pages
            charts_with_page = []
            for page in partition_pages:
                for chart in page.charts:
                    charts_with_page.append(ChartInfoWithPage(
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
                tables=tables_with_page,
                charts=charts_with_page
            )
            partition_detail_list.append(partition_detail)

            logger.debug(f"Partition {partition.partition_id}: {len(tables_with_page)} tables, {len(charts_with_page)} charts")

        # Create PartitionDetails container
        partition_details = PartitionDetails(
            document_id=document.id,
            document_name=document.name,
            total_partitions=len(document.partitions),
            partitions=partition_detail_list
        )

        logger.info(f"Created partition_details for {len(partition_detail_list)} partitions")
        return partition_details

    def _save_partition_details(self, partition_details: PartitionDetails, document_id: str):
        """
        Save partition_details.json file

        Args:
            partition_details: PartitionDetails object
            document_id: Document ID for path
        """
        try:
            doc_dir = self.documents_dir / document_id
            doc_dir.mkdir(parents=True, exist_ok=True)
            output_path = doc_dir / "partition_details.json"

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(partition_details.to_dict(), f, indent=2, ensure_ascii=False)

            logger.info(f"Saved partition_details to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save partition_details: {e}")

    def _build_analysis_prompt(self, context: str = "") -> str:
        """Condensed prompt for faster processing with multi-page content awareness"""
        return f"""Analyze this page and return JSON only:

    {{
    "summary": "brief page summary",
    "tables": [{{"table_id": "table_X_Y", "title": "...", "summary": "..."}}],
    "charts": [{{"chart_id": "chart_X_Y", "title": "...", "chart_type": "line|bar|pie|scatter|area", "summary": "..."}}]
    }}

    Rules:
    - Empty arrays if none found
    - IDs format: table/chart_{{page}}_{{seq}}
    - Return valid JSON only
    - If content appears incomplete/cut-off (table/chart continues on next page), note this in summary with "(continued)" or "(partial)"
    - If content seems to continue from previous page, note with "(continuation from previous page)"
    - For split tables/charts, still create entry but indicate incompleteness

    {context}"""

    def _parse_page_response(self, response: str, page_number: int) -> dict:
        """Parse LLM response to extract structured page data"""
        try:
            # Try to find JSON in the response
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()
            
            # Parse JSON
            data = json.loads(response)
            
            # Validate structure
            if not isinstance(data, dict):
                raise ValueError("Response is not a JSON object")
            
            # Ensure required fields exist
            result = {
                "summary": data.get("summary", "No summary available"),
                "tables": data.get("tables", []),
                "charts": data.get("charts", [])
            }
            
            # Validate tables
            for i, table in enumerate(result["tables"]):
                if "table_id" not in table:
                    table["table_id"] = f"table_{page_number}_{i+1}"
                if "title" not in table:
                    table["title"] = "Untitled Table"
                if "summary" not in table:
                    table["summary"] = ""
            
            # Validate charts
            for i, chart in enumerate(result["charts"]):
                if "chart_id" not in chart:
                    chart["chart_id"] = f"chart_{page_number}_{i+1}"
                if "title" not in chart:
                    chart["title"] = "Untitled Chart"
                if "chart_type" not in chart:
                    chart["chart_type"] = "unknown"
                if "summary" not in chart:
                    chart["summary"] = ""
            
            logger.debug(f"Page {page_number}: Found {len(result['tables'])} tables and {len(result['charts'])} charts")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Response was: {response[:500]}")
            return {
                "summary": response[:500] if response else "Analysis failed",
                "tables": [],
                "charts": []
            }
        except Exception as e:
            logger.error(f"Failed to parse page response: {e}")
            return {
                "summary": "Parsing failed",
                "tables": [],
                "charts": []
            }

    def _save_document_metadata(self, document: Document):
        """
        Save document metadata to metadata.json file.
        Note: partitions array is NOT included in metadata.json for large documents.
        Partition data is stored separately in partition_details.json.
        """
        try:
            doc_dir = self.documents_dir / document.id
            doc_dir.mkdir(parents=True, exist_ok=True)
            output_path = doc_dir / "metadata.json"

            # Pages will include partition_id field (for large documents)
            metadata = {
                "id": document.id,
                "name": document.name,
                "page_count": document.page_count,
                "status": document.status.value,
                "summary": document.summary,
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat(),
                "pages": [p.to_dict() for p in document.pages],
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved complete metadata to: {output_path}")

            self._update_index(document)

        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def _update_index(self, document: Document):
        """Update or create the global document index"""
        try:
            index_path = self.documents_dir / "index.json"
            if index_path.exists():
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
            else:
                index = []

            index_entry = {
                "id": document.id,
                "name": document.name,
                "page_count": document.page_count,
                "status": document.status.value,
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat(),
            }

            index = [e for e in index if e.get("id") != document.id]
            index.append(index_entry)

            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2, ensure_ascii=False)

            logger.debug(f"Updated index at {index_path}")

        except Exception as e:
            logger.error(f"Failed to update index: {e}")

    def load_document(self, document_id: str) -> Optional[Document]:
        """Load previously saved document from metadata.json"""
        try:
            metadata_path = self.documents_dir / document_id / "metadata.json"
            if not metadata_path.exists():
                return None

            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return Document.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load document for {document_id}: {e}")
            return None

    def get_all_documents(self) -> List[Document]:
        """Load all saved documents from metadata.json files"""
        documents = []
        for doc_dir in self.documents_dir.iterdir():
            if not doc_dir.is_dir() or doc_dir.name == "index.json":
                continue

            metadata_file = doc_dir / "metadata.json"
            if not metadata_file.exists():
                continue

            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                document = Document.from_dict(data)
                documents.append(document)
            except Exception as e:
                logger.error(f"Failed to load {metadata_file}: {e}")

        return documents