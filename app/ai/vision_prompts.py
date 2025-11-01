"""
Vision-Specific Prompt Library for GPT-5 Vision
Optimized prompts for different document analysis tasks
"""

from enum import Enum
from typing import Dict, Optional


class AnalysisTask(Enum):
    """Types of vision analysis tasks"""
    GENERAL = "general"
    TABLE_EXTRACTION = "table_extraction"
    TEXT_ANALYSIS = "text_analysis"
    DIAGRAM_UNDERSTANDING = "diagram_understanding"
    FORM_PROCESSING = "form_processing"
    QA = "qa"
    SUMMARY = "summary"


class VisionPromptLibrary:
    """
    Centralized library of vision-optimized prompts for GPT-5
    Each prompt is tailored for specific document analysis tasks
    """
    
    # Core prompts optimized for vision understanding
    PROMPTS: Dict[AnalysisTask, str] = {
        AnalysisTask.GENERAL: """Analyze these document pages and extract all information.
Focus on:
• Text content and structure
• Visual elements (tables, charts, diagrams)
• Relationships between elements
• Key information and context

Provide comprehensive analysis in clear, structured format.""",

        AnalysisTask.TABLE_EXTRACTION: """Extract ALL tables from these pages.
For EACH table:
1. Identify table structure (rows, columns, headers)
2. Extract ALL data accurately preserving relationships
3. Note any merged cells or special formatting
4. Describe the table's purpose

Return tables in clear, structured format with proper alignment.""",

        AnalysisTask.TEXT_ANALYSIS: """Analyze the text content in these pages.
Focus on:
• Main topics and key concepts
• Document structure (headings, sections, paragraphs)
• Important statements and conclusions
• Technical terms and definitions
• Relationships between ideas

Provide clear, well-organized summary of the content.""",

        AnalysisTask.DIAGRAM_UNDERSTANDING: """Analyze ALL diagrams, charts, and visual elements.
For EACH visual element:
1. Describe what it represents
2. Explain relationships shown (arrows, connections, hierarchy)
3. Extract all labels and text
4. Interpret the meaning and purpose
5. Note important details

Provide comprehensive description of visual information.""",

        AnalysisTask.FORM_PROCESSING: """Extract information from forms and structured documents.
Identify:
• Field labels and their values
• Checkboxes, radio buttons (checked/unchecked)
• Signatures, dates, stamps
• Tables and structured data
• Any handwritten content

Return structured data preserving field-value relationships.""",

        AnalysisTask.QA: """Answer the question using information from these pages.
Steps:
1. Locate relevant information
2. Extract exact content that answers the question
3. Provide context if needed
4. Cite specific locations

Answer format:
• Direct answer first
• Supporting details
• Page references
• Related information if relevant

Be precise and cite sources.""",

        AnalysisTask.SUMMARY: """Create a comprehensive summary of these pages.
Include:
• Main topics and themes
• Key points and findings
• Important data (numbers, dates, names)
• Conclusions or recommendations
• Document structure overview

Provide well-organized summary with clear sections."""
    }
    
    @classmethod
    def get_prompt(cls, task: AnalysisTask, custom_context: Optional[str] = None) -> str:
        """
        Get optimized prompt for specific analysis task
        
        Args:
            task: Type of analysis task
            custom_context: Additional context to append to base prompt
            
        Returns:
            Complete prompt optimized for the task
        """
        base_prompt = cls.PROMPTS.get(task, cls.PROMPTS[AnalysisTask.GENERAL])
        
        if custom_context:
            return f"{base_prompt}\n\nAdditional context:\n{custom_context}"
        
        return base_prompt
    
    @classmethod
    def get_combined_image_prompt(
        cls,
        task: AnalysisTask,
        num_pages: int,
        grid_description: str,
        custom_context: Optional[str] = None
    ) -> str:
        """
        Get prompt for combined image analysis
        
        Args:
            task: Type of analysis task
            num_pages: Number of pages in combined image
            grid_description: Simple grid layout description
            custom_context: Additional context
            
        Returns:
            Complete prompt for combined image analysis
        """
        base_prompt = cls.get_prompt(task, custom_context)
        
        header = f"""You are analyzing a combined image containing {num_pages} document page(s).

{grid_description}

"""
        return header + base_prompt
    
    @classmethod
    def get_qa_prompt(cls, question: str, num_pages: int, grid_description: str) -> str:
        """
        Get optimized prompt for Q&A task
        
        Args:
            question: User's question
            num_pages: Number of pages
            grid_description: Grid layout description
            
        Returns:
            Complete Q&A prompt
        """
        header = f"""You are analyzing {num_pages} document page(s) to answer a question.

{grid_description}

QUESTION: {question}

"""
        
        qa_instructions = """Answer the question using information from the pages above.

Requirements:
✓ Provide direct, accurate answer
✓ Quote relevant text when possible
✓ Cite specific page numbers
✓ Include supporting context
✓ Use clear, well-formatted response

If information is not found, state that clearly."""
        
        return header + qa_instructions
    
    @classmethod
    def get_summary_prompt(cls, num_pages: int, grid_description: str, focus_areas: Optional[str] = None) -> str:
        """
        Get optimized prompt for summary generation
        
        Args:
            num_pages: Number of pages
            grid_description: Grid layout description
            focus_areas: Specific areas to focus on
            
        Returns:
            Complete summary prompt
        """
        header = f"""You are summarizing {num_pages} document page(s).

{grid_description}

"""
        
        summary_instructions = """Create a comprehensive, well-structured summary.

Include:
• Main topics and key concepts
• Important data and findings
• Conclusions and recommendations
• Document organization

Format:
- Use clear headings and sections
- Bullet points for key information
- Proper markdown formatting
- Logical flow and structure

"""
        
        if focus_areas:
            summary_instructions += f"\nFocus especially on: {focus_areas}\n"
        
        return header + summary_instructions
    
    @classmethod
    def detect_task_from_question(cls, question: str) -> AnalysisTask:
        """
        Automatically detect the most appropriate task type from a question
        
        Args:
            question: User's question or request
            
        Returns:
            Most suitable analysis task
        """
        question_lower = question.lower()
        
        # Table-related keywords
        if any(keyword in question_lower for keyword in [
            'table', 'tabular', 'data', 'column', 'row', 'cell'
        ]):
            return AnalysisTask.TABLE_EXTRACTION
        
        # Diagram-related keywords
        if any(keyword in question_lower for keyword in [
            'diagram', 'chart', 'graph', 'figure', 'illustration', 'visual', 'image'
        ]):
            return AnalysisTask.DIAGRAM_UNDERSTANDING
        
        # Form-related keywords
        if any(keyword in question_lower for keyword in [
            'form', 'field', 'checkbox', 'signature', 'date', 'fill'
        ]):
            return AnalysisTask.FORM_PROCESSING
        
        # Summary-related keywords
        if any(keyword in question_lower for keyword in [
            'summary', 'summarize', 'overview', 'main point', 'key point', 'recap'
        ]):
            return AnalysisTask.SUMMARY
        
        # Text analysis keywords
        if any(keyword in question_lower for keyword in [
            'explain', 'describe', 'discuss', 'analyze', 'interpret'
        ]):
            return AnalysisTask.TEXT_ANALYSIS
        
        # Default to Q&A for questions
        if '?' in question or any(keyword in question_lower for keyword in [
            'what', 'where', 'when', 'who', 'why', 'how', 'which'
        ]):
            return AnalysisTask.QA
        
        # Default to general analysis
        return AnalysisTask.GENERAL
