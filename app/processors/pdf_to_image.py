import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
import os
import json
import hashlib
from typing import Optional, List
from datetime import datetime, timezone
from loguru import logger

from document import (
    Document,
    Page,
    DocumentStatus,
    Partition,
    PartitionDetail,
    PartitionDetails,
    TableInfoWithPage,
    ChartInfoWithPage,
)


class VisionPDFProcessor:
    """PDF processor that converts to images with auto-partitioning for large docs"""

    def __init__(
        self,
        render_scale: float = 1.5,
        jpeg_quality: int = 75,
        max_image_size: tuple = (1024, 1024),
        storage_root: str = None,
        partition_size: int = 20,  # Số trang mỗi partition
        partition_overlap: int = 0  # Số trang overlap giữa các partition (nếu cần)
    ):
        if storage_root is None:
            storage_root = os.environ.get("FLEX_RAG_DATA_LOCATION", "../../flex_rag_data_location")
    
        self.storage_root = Path(storage_root)
        self.render_scale = render_scale
        self.jpeg_quality = jpeg_quality
        self.max_image_size = max_image_size
        self.partition_size = partition_size
        self.partition_overlap = partition_overlap
        self.documents_dir = self.storage_root / "documents"
        

        # Ensure directories exist
        self.documents_dir.mkdir(parents=True, exist_ok=True)

    def _generate_doc_id(self, file_path: str) -> str:
        """Generate unique document ID from file path and timestamp"""
        content = f"{file_path}_{datetime.now(timezone.utc).isoformat()}"
        return f"doc_{hashlib.md5(content.encode()).hexdigest()[:12]}"

    def _create_document_directory(self, doc_id: str) -> Path:
        """Create directory structure for a document"""
        doc_dir = self.documents_dir / doc_id
        pages_dir = doc_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        return doc_dir

    def _create_partitions(self, total_pages: int) -> List[Partition]:
        """
        Tạo partitions tự động dựa trên số trang
        
        Args:
            total_pages: Tổng số trang của tài liệu
            
        Returns:
            List of Partition objects
        """
        if total_pages <= self.partition_size:
            return []
        
        partitions = []
        partition_id = 1
        start_page = 1
        
        while start_page <= total_pages:
            # Tính end_page cho partition hiện tại
            end_page = min(start_page + self.partition_size - 1, total_pages)
            
            # Tạo partition
            partition = Partition(
                partition_id=partition_id,
                page_range=(start_page, end_page),
                summary=""  # Sẽ được điền sau khi phân tích
            )
            partitions.append(partition)
            
            logger.info(f"Created partition {partition_id}: pages {start_page}-{end_page}")
            
            # Move to next partition (trừ overlap nếu có)
            start_page = end_page + 1 - self.partition_overlap
            partition_id += 1
        
        return partitions

    def _assign_pages_to_partitions(
        self, 
        pages: List[Page], 
        partitions: List[Partition]
    ) -> None:
        """
        Gán partition_id cho từng page
        
        Args:
            pages: List of Page objects
            partitions: List of Partition objects
        """
        for page in pages:
            for partition in partitions:
                start, end = partition.page_range
                if start <= page.page_number <= end:
                    page.partition_id = partition.partition_id
                    break

    def _save_metadata(self, document: Document, doc_dir: Path) -> None:
        """
        Lưu metadata.json cho document
        
        Args:
            document: Document object
            doc_dir: Document directory path
        """
        metadata_path = doc_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(document.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata to {metadata_path}")

    def _save_partition_details(
        self, 
        document: Document, 
        doc_dir: Path
    ) -> None:
        """
        Tạo và lưu partition_summary.json cho tài liệu lớn
        
        Args:
            document: Document object
            doc_dir: Document directory path
        """
        if not document.has_partitions():
            return
        
        # Tạo PartitionDetails
        partition_details_list = []
        
        for partition in document.partitions:
            # Lấy tất cả pages thuộc partition này
            partition_pages = [
                p for p in document.pages 
                if p.partition_id == partition.partition_id
            ]
            
            # Aggregate tables và charts
            tables = []
            charts = []
            
            for page in partition_pages:
                for table in page.tables:
                    tables.append(TableInfoWithPage(
                        table_id=table.table_id,
                        page_number=page.page_number,
                        title=table.title,
                        summary=table.summary
                    ))
                
                for chart in page.charts:
                    charts.append(ChartInfoWithPage(
                        chart_id=chart.chart_id,
                        page_number=page.page_number,
                        title=chart.title,
                        chart_type=chart.chart_type,
                        summary=chart.summary
                    ))
            
            # Tạo PartitionDetail
            partition_detail = PartitionDetail(
                partition_id=partition.partition_id,
                page_range=partition.page_range,
                page_count=partition.get_page_count(),
                summary=partition.summary,
                tables=tables,
                charts=charts
            )
            partition_details_list.append(partition_detail)
        
        # Tạo PartitionDetails container
        partition_details = PartitionDetails(
            document_id=document.id,
            document_name=document.name,
            total_partitions=len(document.partitions),
            partitions=partition_details_list
        )
        
        # Lưu file
        details_path = doc_dir / "partition_summary.json"
        with open(details_path, 'w', encoding='utf-8') as f:
            json.dump(partition_details.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved partition details to {details_path}")
        logger.info(f"Total partitions: {len(partition_details_list)}")

    async def process(self, file_path: str, doc_id: Optional[str] = None) -> Document:
        """
        Convert PDF pages to JPEG images and save in structured directory.
        Automatically creates partitions for documents >20 pages.

        Args:
            file_path: Path to the PDF file
            doc_id: Optional document ID (generated if not provided)

        Returns:
            Document object with all metadata
        """
        try:
            logger.info(f"Processing PDF with vision: {file_path}")

            # Generate document ID if not provided
            if not doc_id:
                doc_id = self._generate_doc_id(file_path)

            # Create document directory structure
            doc_dir = self._create_document_directory(doc_id)
            pages_dir = doc_dir / "pages"

            # Open PDF with PyMuPDF
            pdf = fitz.open(file_path)
            total_pages = len(pdf)
            
            logger.info(f"Document has {total_pages} pages")
            
            # Tạo partitions nếu cần (>20 pages)
            partitions = []
            if total_pages > self.partition_size:
                logger.info(f"Large document detected. Creating partitions...")
                partitions = self._create_partitions(total_pages)
                logger.info(f"Created {len(partitions)} partitions")
            else:
                logger.info("Document has ≤20 pages. No partitioning needed.")

            # Process pages
            pages = []
            for page_num in range(total_pages):
                page = pdf[page_num]

                # Render page to high-quality pixmap
                mat = fitz.Matrix(self.render_scale, self.render_scale)
                pix = page.get_pixmap(matrix=mat, alpha=False)

                # Convert to PIL Image
                img = Image.frombytes(
                    "RGB",
                    [pix.width, pix.height],
                    pix.samples
                )

                # Resize if exceeds max size
                if (img.width > self.max_image_size[0] or
                    img.height > self.max_image_size[1]):
                    img.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
                    logger.debug(f"Resized page {page_num+1} to {img.size}")

                # Save as optimized JPEG in pages directory
                output_path = pages_dir / f"page_{page_num + 1}.jpg"
                img.save(
                    output_path,
                    "JPEG",
                    quality=self.jpeg_quality,
                    optimize=True
                )

                # Create Page object with relative path
                pages.append(Page(
                    page_number=page_num + 1,
                    image_path=str(output_path),
                    width=img.width,
                    height=img.height
                ))

                if (page_num + 1) % 10 == 0:
                    logger.info(f"Processed {page_num + 1}/{total_pages} pages")

            pdf.close()

            # Assign partition_id to pages if partitions exist
            if partitions:
                self._assign_pages_to_partitions(pages, partitions)
                logger.info("Assigned pages to partitions")

            # Create Document object
            document = Document(
                id=doc_id,
                name=Path(file_path).name,
                page_count=len(pages),
                pages=pages,
                partitions=partitions,
                status=DocumentStatus.READY,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )

            # Save metadata
            self._save_metadata(document, doc_dir)
            
            # Save partition details if needed
            if document.has_partitions():
                self._save_partition_details(document, doc_dir)

            logger.info(f"✅ Successfully processed {len(pages)} pages from {file_path}")
            logger.info(f"📁 Document directory: {doc_dir}")
            if document.has_partitions():
                logger.info(f"📑 Created {len(partitions)} partitions")

            return document

        except Exception as e:
            logger.error(f"❌ Failed to process PDF: {e}")
            raise
    
    def supports(self, file_path: str) -> bool:
        """Check if file is PDF"""
        return file_path.lower().endswith('.pdf')
