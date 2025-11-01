"""
Specialized service for defect detection in documents using GPT-4/GPT-5 vision capabilities
Optimized for high-quality visual inspection and defect classification
"""
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger
from app.models.document import Document, Page
from app.providers.base import BaseProvider
from app.ai.vision_analysis import VisionAnalysisService


class DefectDetectionService(VisionAnalysisService):
    """
    Enhanced service specifically for defect detection in PDF documents
    
    Features:
    - Specialized prompts for visual, text, and structural defects
    - Severity classification (CRITICAL, HIGH, MEDIUM, LOW)
    - Quality scoring and comprehensive reporting
    - Batch processing with parallel execution
    - Cost-optimized two-phase analysis
    """
    
    def __init__(self, provider: BaseProvider, storage_root: str = None):
        """
        Initialize defect detection service
        
        Args:
            provider: LLM provider with vision capabilities (GPT-4/GPT-5)
            storage_root: Root directory for storing analysis results
        """
        super().__init__(provider, storage_root)
        
        # Defect tracking
        self.total_defects_found = 0
        self.pages_analyzed = 0
        self.total_detection_cost = 0.0
        
        logger.info("DefectDetectionService initialized for high-precision defect analysis")
    
    async def detect_defects_in_document(
        self,
        document: Document,
        severity_filter: Optional[List[str]] = None,
        defect_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive defect detection across entire document
        
        Args:
            document: Document to analyze for defects
            severity_filter: Only return defects matching severity levels 
                           (e.g., ["CRITICAL", "HIGH"])
            defect_types: Filter by defect types (e.g., ["visual", "structural"])
            
        Returns:
            Detailed defect report with:
            - Total defects found
            - Severity breakdown
            - Type breakdown
            - Quality score (0-100)
            - Detailed defect list per page
        """
        try:
            logger.info(f"🔍 Starting defect detection: {document.name}")
            logger.info(f"   Pages to analyze: {document.page_count}")
            
            all_defects = []
            defect_pages = []
            total_defects = 0
            analysis_cost = 0.0
            
            # Group pages by combined image for efficient batch processing
            image_to_pages: Dict[str, List[Page]] = {}
            for page in document.pages:
                image_path = page.image_path
                if image_path not in image_to_pages:
                    image_to_pages[image_path] = []
                image_to_pages[image_path].append(page)
            
            # Analyze each combined image
            for image_path, pages_in_image in image_to_pages.items():
                logger.info(f"Analyzing {len(pages_in_image)} pages in: {Path(image_path).name}")
                
                # Detect defects in combined image
                page_defects_list = await self._detect_defects_in_combined_image(
                    image_path=image_path,
                    pages_in_image=pages_in_image,
                    document_name=document.name
                )
                
                # Process results
                for page_defects in page_defects_list:
                    if page_defects.get("defects_found", False):
                        defect_list = page_defects.get("defects", [])
                        all_defects.extend(defect_list)
                        defect_pages.append(page_defects["page_number"])
                        total_defects += len(defect_list)
                        self.pages_analyzed += 1
                
                # Track cost
                cost = self.provider.get_last_cost() or 0.0
                analysis_cost += cost
            
            # Generate comprehensive defect report
            report = self._generate_defect_report(
                document=document,
                all_defects=all_defects,
                defect_pages=defect_pages,
                severity_filter=severity_filter,
                defect_types=defect_types,
                analysis_cost=analysis_cost
            )
            
            self.total_defects_found += total_defects
            self.total_detection_cost += analysis_cost
            
            logger.info(f"✅ Defect detection complete!")
            logger.info(f"   Total defects: {total_defects}")
            logger.info(f"   Pages with defects: {len(defect_pages)}/{document.page_count}")
            logger.info(f"   Quality score: {report['quality_score']}/100")
            logger.info(f"   Cost: ${analysis_cost:.4f}")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to detect defects in document: {e}")
            raise
    
    async def _detect_defects_in_combined_image(
        self,
        image_path: str,
        pages_in_image: List[Page],
        document_name: str
    ) -> List[Dict[str, Any]]:
        """
        Detect defects in a combined image containing multiple pages
        
        Args:
            image_path: Path to combined image
            pages_in_image: List of pages in this image
            document_name: Name of the document
            
        Returns:
            List of defect analysis per page
        """
        try:
            # Build specialized defect detection prompt
            prompt = self._build_defect_detection_prompt(pages_in_image, document_name)
            
            # Prepare multimodal message
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert quality control inspector with advanced visual defect detection capabilities. Your task is to meticulously analyze document pages for any defects, anomalies, or quality issues. Provide precise, structured analysis with confidence levels and severity ratings."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_path", "image_path": image_path, "detail": "high"},
                    ],
                },
            ]
            
            # Get defect analysis from vision model
            response = await self.provider.process_multimodal_messages(
                messages=messages,
                max_tokens=4000,  # Higher limit for detailed defect descriptions
                temperature=1.0   # GPT-5 requirement
            )
            
            # Parse response to extract defect data
            defect_analyses = self._parse_defect_response(response, pages_in_image)
            
            return defect_analyses
            
        except Exception as e:
            logger.error(f"Failed to detect defects in combined image: {e}")
            # Return empty defect analysis
            return [
                {
                    "page_number": page.page_number,
                    "defects_found": False,
                    "defect_count": 0,
                    "defects": [],
                    "summary": f"Defect analysis failed: {str(e)}",
                    "quality_status": "UNKNOWN"
                }
                for page in pages_in_image
            ]
    
    def _build_defect_detection_prompt(
        self, 
        pages_in_image: List[Page],
        document_name: str
    ) -> str:
        """Build specialized prompt for defect detection"""
        page_numbers = [str(p.page_number) for p in pages_in_image]
        page_list = ", ".join(page_numbers)
        
        # Build grid layout description
        grid_layout = self._build_grid_layout_description(pages_in_image)
        
        return f"""DEFECT DETECTION ANALYSIS
Document: {document_name}
Pages to analyze: {page_list}

{grid_layout}

CRITICAL DEFECT DETECTION INSTRUCTIONS:

You are performing a comprehensive quality inspection. Analyze each page systematically for defects.

🔍 DEFECT CATEGORIES TO DETECT:

1. **VISUAL DEFECTS** (Physical/Visual Issues):
   - Scratches, cracks, dents, or surface damage
   - Discoloration, stains, or contamination marks
   - Blurriness, distortion, or poor image quality
   - Misalignment or skewed content
   - Missing components, parts, or visual elements
   - Color inconsistencies or fading
   - Artifacts or compression issues

2. **TEXT/DATA DEFECTS** (Content Issues):
   - Typos, spelling errors, or grammatical mistakes
   - Missing text, labels, or information
   - Illegible text or poor OCR quality
   - Wrong dates, numbers, values, or measurements
   - Inconsistent formatting or fonts
   - Truncated or cut-off text
   - Duplicate or conflicting information

3. **STRUCTURAL DEFECTS** (Document Structure):
   - Deformations, warping, or bending
   - Tears, breaks, or separations
   - Page orientation issues
   - Missing pages or sections
   - Incorrect page ordering
   - Border or margin problems

4. **QUALITY DEFECTS** (Overall Quality):
   - Low resolution or pixelation
   - Poor contrast or brightness
   - Watermarks obscuring content
   - Overlapping elements
   - Incomplete rendering

📊 SEVERITY CLASSIFICATION:
- **CRITICAL**: Makes document unusable or misleading (e.g., wrong data, illegible content)
- **HIGH**: Significantly impacts usability or understanding (e.g., major visual issues)
- **MEDIUM**: Noticeable but doesn't prevent usage (e.g., minor formatting issues)
- **LOW**: Cosmetic issues with minimal impact (e.g., slight discoloration)
- **NONE**: No defects detected, page is acceptable

🎯 CONFIDENCE LEVELS:
- **HIGH**: Defect is clearly visible and unambiguous
- **MEDIUM**: Defect is likely but could be interpretation-dependent
- **LOW**: Possible defect requiring verification

📝 OUTPUT FORMAT:
For EACH page in the grid, provide a JSON object:

{{
  "page_number": <page_number>,
  "defects_found": true/false,
  "defect_count": <number>,
  "quality_status": "EXCELLENT|GOOD|ACCEPTABLE|POOR|CRITICAL",
  "defects": [
    {{
      "type": "visual|text|structural|quality",
      "severity": "CRITICAL|HIGH|MEDIUM|LOW",
      "description": "Precise description of the defect",
      "location": "top-left|center|bottom-right|entire-page|etc",
      "confidence": "HIGH|MEDIUM|LOW",
      "recommendation": "Suggested action to fix"
    }}
  ],
  "summary": "Brief overall assessment of this page",
  "notes": "Any additional observations"
}}

Return as a JSON array containing one object per page:
[
  {{ page 1 analysis }},
  {{ page 2 analysis }},
  ...
]

⚠️ IMPORTANT:
- Be thorough but precise - don't over-report minor issues
- Base analysis ONLY on what you can see in the image
- If a page appears perfect, report defects_found: false
- For text defects, verify by reading the actual content visible
- Use the grid position mapping above to identify correct pages
- Provide actionable recommendations for fixing defects

Begin your detailed defect analysis now:"""
    
    def _parse_defect_response(
        self, 
        response: str, 
        pages_in_image: List[Page]
    ) -> List[Dict[str, Any]]:
        """Parse LLM response into structured defect data"""
        try:
            # Clean the response to extract JSON array
            response = response.strip()
            
            # Find JSON array in response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                defect_analyses = json.loads(json_str)
                
                # Validate and clean the analyses
                valid_analyses = []
                for analysis in defect_analyses:
                    if isinstance(analysis, dict) and "page_number" in analysis:
                        # Ensure all required fields
                        clean_analysis = {
                            "page_number": int(analysis.get("page_number", 0)),
                            "defects_found": bool(analysis.get("defects_found", False)),
                            "defect_count": int(analysis.get("defect_count", 0)),
                            "quality_status": str(analysis.get("quality_status", "UNKNOWN")),
                            "defects": analysis.get("defects", []),
                            "summary": str(analysis.get("summary", "No summary available")),
                            "notes": str(analysis.get("notes", ""))
                        }
                        valid_analyses.append(clean_analysis)
                
                # Ensure we have analysis for all expected pages
                expected_pages = {p.page_number for p in pages_in_image}
                found_pages = {a["page_number"] for a in valid_analyses}
                
                # Add missing pages with default analysis
                for page in pages_in_image:
                    if page.page_number not in found_pages:
                        valid_analyses.append({
                            "page_number": page.page_number,
                            "defects_found": False,
                            "defect_count": 0,
                            "quality_status": "UNKNOWN",
                            "defects": [],
                            "summary": "Analysis not available for this page",
                            "notes": ""
                        })
                
                return valid_analyses
            
            else:
                raise ValueError("No valid JSON array found in response")
                
        except Exception as e:
            logger.error(f"Failed to parse defect response: {e}")
            logger.debug(f"Response was: {response[:500]}...")
            
            # Return default analysis for all pages
            return [
                {
                    "page_number": page.page_number,
                    "defects_found": False,
                    "defect_count": 0,
                    "quality_status": "UNKNOWN",
                    "defects": [],
                    "summary": "Failed to parse defect analysis response",
                    "notes": str(e)
                }
                for page in pages_in_image
            ]
    
    def _generate_defect_report(
        self,
        document: Document,
        all_defects: List[Dict],
        defect_pages: List[int],
        severity_filter: Optional[List[str]],
        defect_types: Optional[List[str]],
        analysis_cost: float
    ) -> Dict[str, Any]:
        """Generate comprehensive defect report"""
        
        # Filter by severity if specified
        filtered_defects = all_defects
        if severity_filter:
            filtered_defects = [
                d for d in filtered_defects 
                if d.get("severity") in severity_filter
            ]
        
        # Filter by type if specified
        if defect_types:
            filtered_defects = [
                d for d in filtered_defects 
                if d.get("type") in defect_types
            ]
        
        # Count by severity
        severity_counts = {
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0
        }
        
        for defect in all_defects:  # Use all_defects for statistics
            severity = defect.get("severity", "UNKNOWN")
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        # Count by type
        type_counts = {}
        for defect in all_defects:
            defect_type = defect.get("type", "unknown")
            type_counts[defect_type] = type_counts.get(defect_type, 0) + 1
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            total_pages=document.page_count,
            total_defects=len(all_defects),
            severity_counts=severity_counts
        )
        
        # Determine overall status
        overall_status = self._determine_overall_status(severity_counts, quality_score)
        
        # Group defects by page for detailed reporting
        defects_by_page = {}
        for defect in filtered_defects:
            # Find which page this defect belongs to
            for page_num in defect_pages:
                if page_num not in defects_by_page:
                    defects_by_page[page_num] = []
        
        # Re-parse to group properly
        for page_num in defect_pages:
            page_defects = [d for d in filtered_defects if d.get("page_number") == page_num]
            if page_defects or page_num in defect_pages:
                defects_by_page[page_num] = page_defects
        
        return {
            "document_name": document.name,
            "document_id": document.id,
            "total_pages": document.page_count,
            "pages_analyzed": document.page_count,
            "pages_with_defects": len(defect_pages),
            "defect_page_numbers": sorted(defect_pages),
            "clean_pages": document.page_count - len(defect_pages),
            "total_defects": len(all_defects),
            "filtered_defects": len(filtered_defects),
            "severity_breakdown": severity_counts,
            "type_breakdown": type_counts,
            "quality_score": quality_score,
            "overall_status": overall_status,
            "defects": filtered_defects,
            "defects_by_page": defects_by_page,
            "analysis_cost": analysis_cost,
            "filters_applied": {
                "severity": severity_filter,
                "types": defect_types
            },
            "recommendations": self._generate_recommendations(severity_counts, type_counts)
        }
    
    def _calculate_quality_score(
        self,
        total_pages: int,
        total_defects: int,
        severity_counts: Dict[str, int]
    ) -> float:
        """
        Calculate overall quality score (0-100)
        
        Scoring formula:
        - Start with 100
        - Deduct weighted points based on defect severity
        - CRITICAL defects: -10 points each
        - HIGH defects: -5 points each
        - MEDIUM defects: -2 points each
        - LOW defects: -1 point each
        """
        
        # Weight defects by severity
        weighted_defects = (
            severity_counts.get("CRITICAL", 0) * 10 +
            severity_counts.get("HIGH", 0) * 5 +
            severity_counts.get("MEDIUM", 0) * 2 +
            severity_counts.get("LOW", 0) * 1
        )
        
        # Calculate max possible weighted defects (assume all pages could have critical defects)
        max_weighted_defects = total_pages * 10
        
        # Calculate score
        if max_weighted_defects > 0:
            quality_score = max(0, 100 - (weighted_defects / max_weighted_defects * 100))
        else:
            quality_score = 100
        
        return round(quality_score, 2)
    
    def _determine_overall_status(
        self, 
        severity_counts: Dict[str, int],
        quality_score: float
    ) -> str:
        """Determine overall document status based on defects and quality score"""
        
        if severity_counts.get("CRITICAL", 0) > 0:
            return "CRITICAL - Requires immediate attention"
        elif severity_counts.get("HIGH", 0) >= 3:
            return "HIGH PRIORITY - Multiple serious defects found"
        elif severity_counts.get("HIGH", 0) > 0:
            return "NEEDS REVIEW - Serious defects present"
        elif quality_score >= 90:
            return "EXCELLENT - Minimal or no defects"
        elif quality_score >= 75:
            return "GOOD - Minor defects only"
        elif quality_score >= 60:
            return "ACCEPTABLE - Some defects present"
        else:
            return "POOR - Multiple defects affecting quality"
    
    def _generate_recommendations(
        self,
        severity_counts: Dict[str, int],
        type_counts: Dict[str, int]
    ) -> List[str]:
        """Generate actionable recommendations based on defect analysis"""
        recommendations = []
        
        # Severity-based recommendations
        if severity_counts.get("CRITICAL", 0) > 0:
            recommendations.append(
                f"⚠️ URGENT: {severity_counts['CRITICAL']} critical defect(s) found. "
                "Document may be unusable or contain misleading information. Immediate review required."
            )
        
        if severity_counts.get("HIGH", 0) > 0:
            recommendations.append(
                f"🔴 {severity_counts['HIGH']} high-severity defect(s) detected. "
                "These significantly impact document quality and should be addressed promptly."
            )
        
        if severity_counts.get("MEDIUM", 0) >= 5:
            recommendations.append(
                f"🟡 Multiple medium-severity defects ({severity_counts['MEDIUM']}) found. "
                "Consider batch correction to improve overall quality."
            )
        
        # Type-based recommendations
        if type_counts.get("text", 0) > 0:
            recommendations.append(
                f"📝 {type_counts['text']} text/data defect(s) found. "
                "Recommend OCR re-processing or manual text correction."
            )
        
        if type_counts.get("visual", 0) > 0:
            recommendations.append(
                f"👁️ {type_counts['visual']} visual defect(s) detected. "
                "Consider re-scanning with higher quality settings or cleaning source documents."
            )
        
        if type_counts.get("structural", 0) > 0:
            recommendations.append(
                f"🏗️ {type_counts['structural']} structural defect(s) found. "
                "Review document structure and page alignment."
            )
        
        if type_counts.get("quality", 0) > 0:
            recommendations.append(
                f"⚙️ {type_counts['quality']} quality issue(s) identified. "
                "Improve scanning/capture settings or image processing."
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append("✅ No significant defects found. Document quality is satisfactory.")
        
        return recommendations
    
    async def batch_detect_defects(
        self,
        documents: List[Document],
        parallel_limit: int = 3,
        severity_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple documents in parallel with rate limiting
        
        Args:
            documents: List of documents to analyze
            parallel_limit: Maximum concurrent analyses
            severity_filter: Optional severity filter
            
        Returns:
            List of defect reports, one per document
        """
        import asyncio
        
        logger.info(f"Starting batch defect detection: {len(documents)} documents")
        logger.info(f"Parallel limit: {parallel_limit}")
        
        semaphore = asyncio.Semaphore(parallel_limit)
        
        async def process_with_limit(doc):
            async with semaphore:
                logger.info(f"Processing: {doc.name}")
                return await self.detect_defects_in_document(
                    document=doc,
                    severity_filter=severity_filter
                )
        
        results = await asyncio.gather(
            *[process_with_limit(doc) for doc in documents],
            return_exceptions=True
        )
        
        # Handle exceptions in results
        processed_results = []
        for doc, result in zip(documents, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process {doc.name}: {result}")
                processed_results.append({
                    "document_name": doc.name,
                    "error": str(result),
                    "status": "FAILED"
                })
            else:
                processed_results.append(result)
        
        logger.info(f"✅ Batch processing complete: {len(documents)} documents")
        logger.info(f"   Total cost: ${self.total_detection_cost:.4f}")
        logger.info(f"   Total defects found: {self.total_defects_found}")
        
        return processed_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get defect detection statistics for this session"""
        return {
            "pages_analyzed": self.pages_analyzed,
            "total_defects_found": self.total_defects_found,
            "total_cost": self.total_detection_cost,
            "average_cost_per_page": (
                self.total_detection_cost / self.pages_analyzed 
                if self.pages_analyzed > 0 else 0
            ),
            "defects_per_page": (
                self.total_defects_found / self.pages_analyzed 
                if self.pages_analyzed > 0 else 0
            )
        }
    
    def reset_statistics(self):
        """Reset defect detection statistics"""
        self.total_defects_found = 0
        self.pages_analyzed = 0
        self.total_detection_cost = 0.0
        logger.info("Defect detection statistics reset")
