"""
Dynamic Token Budget Optimizer
Intelligently allocates tokens based on content complexity and task requirements
"""

from enum import Enum
from typing import Dict, Optional


class ContentComplexity(Enum):
    """Content complexity levels"""
    SIMPLE = "simple"          # Plain text, simple structure
    MODERATE = "moderate"      # Mixed content, some structure
    COMPLEX = "complex"        # Tables, diagrams, dense content
    VERY_COMPLEX = "very_complex"  # Technical docs, multiple elements


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = "low"           # Less critical tasks
    NORMAL = "normal"     # Standard tasks
    HIGH = "high"         # Important tasks requiring detail
    CRITICAL = "critical" # Must have maximum quality


class TokenBudgetOptimizer:
    """
    Dynamically optimizes token allocation for vision and text analysis
    Balances quality with cost efficiency
    """
    
    # Base token budgets for different scenarios
    BASE_BUDGETS = {
        "vision_single_page": 2000,
        "vision_combined_2_4": 3000,
        "vision_combined_5_9": 4000,
        "text_fallback": 1500,
        "page_selection": 500,
        "summary": 800,
    }
    
    # Complexity multipliers
    COMPLEXITY_MULTIPLIERS = {
        ContentComplexity.SIMPLE: 0.7,
        ContentComplexity.MODERATE: 1.0,
        ContentComplexity.COMPLEX: 1.3,
        ContentComplexity.VERY_COMPLEX: 1.5,
    }
    
    # Priority multipliers
    PRIORITY_MULTIPLIERS = {
        TaskPriority.LOW: 0.8,
        TaskPriority.NORMAL: 1.0,
        TaskPriority.HIGH: 1.2,
        TaskPriority.CRITICAL: 1.5,
    }
    
    @classmethod
    def calculate_vision_budget(
        cls,
        num_pages: int,
        complexity: ContentComplexity = ContentComplexity.MODERATE,
        priority: TaskPriority = TaskPriority.NORMAL,
        has_tables: bool = False,
        has_diagrams: bool = False
    ) -> int:
        """
        Calculate optimal token budget for vision analysis
        
        Args:
            num_pages: Number of pages being analyzed
            complexity: Estimated content complexity
            priority: Task priority level
            has_tables: Whether content includes tables
            has_diagrams: Whether content includes diagrams
            
        Returns:
            Optimized token budget
        """
        # Get base budget based on page count
        if num_pages == 1:
            base = cls.BASE_BUDGETS["vision_single_page"]
        elif num_pages <= 4:
            base = cls.BASE_BUDGETS["vision_combined_2_4"]
        else:
            base = cls.BASE_BUDGETS["vision_combined_5_9"]
        
        # Apply complexity multiplier
        complexity_factor = cls.COMPLEXITY_MULTIPLIERS[complexity]
        budget = int(base * complexity_factor)
        
        # Apply priority multiplier
        priority_factor = cls.PRIORITY_MULTIPLIERS[priority]
        budget = int(budget * priority_factor)
        
        # Adjust for special content types
        if has_tables:
            budget = int(budget * 1.2)  # Tables need more tokens
        
        if has_diagrams:
            budget = int(budget * 1.15)  # Diagrams need more description
        
        # Ensure reasonable bounds
        min_budget = 1000
        max_budget = 6000  # Keep costs reasonable
        
        return max(min_budget, min(budget, max_budget))
    
    @classmethod
    def calculate_text_budget(
        cls,
        text_length: int,
        complexity: ContentComplexity = ContentComplexity.MODERATE,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> int:
        """
        Calculate optimal token budget for text-based analysis
        
        Args:
            text_length: Length of input text (characters)
            complexity: Content complexity
            priority: Task priority
            
        Returns:
            Optimized token budget
        """
        base = cls.BASE_BUDGETS["text_fallback"]
        
        # Adjust based on input length
        if text_length < 1000:
            base = int(base * 0.7)  # Short text needs less
        elif text_length > 5000:
            base = int(base * 1.3)  # Long text may need more
        
        # Apply complexity and priority
        complexity_factor = cls.COMPLEXITY_MULTIPLIERS[complexity]
        priority_factor = cls.PRIORITY_MULTIPLIERS[priority]
        
        budget = int(base * complexity_factor * priority_factor)
        
        # Bounds
        min_budget = 500
        max_budget = 3000
        
        return max(min_budget, min(budget, max_budget))
    
    @classmethod
    def calculate_page_selection_budget(
        cls,
        num_available_pages: int,
        summary_length: int,
        complexity: ContentComplexity = ContentComplexity.MODERATE
    ) -> int:
        """
        Calculate optimal budget for page selection task
        
        Args:
            num_available_pages: Number of pages to choose from
            summary_length: Total length of summaries (characters)
            complexity: Query complexity
            
        Returns:
            Optimized token budget
        """
        base = cls.BASE_BUDGETS["page_selection"]
        
        # Adjust for number of pages (more pages = harder decision)
        if num_available_pages > 20:
            base = int(base * 1.3)
        elif num_available_pages < 5:
            base = int(base * 0.8)
        
        # Adjust for summary length
        if summary_length > 10000:
            base = int(base * 1.2)  # Long summaries need more reasoning
        
        # Apply complexity
        complexity_factor = cls.COMPLEXITY_MULTIPLIERS[complexity]
        budget = int(base * complexity_factor)
        
        # Bounds
        min_budget = 300
        max_budget = 800
        
        return max(min_budget, min(budget, max_budget))
    
    @classmethod
    def detect_complexity(cls, content_indicators: Dict[str, any]) -> ContentComplexity:
        """
        Automatically detect content complexity from indicators
        
        Args:
            content_indicators: Dictionary with complexity signals
                - has_tables: bool
                - has_diagrams: bool
                - has_code: bool
                - text_density: float (0-1)
                - num_sections: int
                - technical_terms: int
                
        Returns:
            Estimated complexity level
        """
        score = 0
        
        # Check various complexity indicators
        if content_indicators.get("has_tables"):
            score += 2
        
        if content_indicators.get("has_diagrams"):
            score += 2
        
        if content_indicators.get("has_code"):
            score += 2
        
        text_density = content_indicators.get("text_density", 0.5)
        if text_density > 0.8:
            score += 1
        
        num_sections = content_indicators.get("num_sections", 0)
        if num_sections > 10:
            score += 1
        
        technical_terms = content_indicators.get("technical_terms", 0)
        if technical_terms > 20:
            score += 2
        
        # Map score to complexity
        if score <= 2:
            return ContentComplexity.SIMPLE
        elif score <= 4:
            return ContentComplexity.MODERATE
        elif score <= 6:
            return ContentComplexity.COMPLEX
        else:
            return ContentComplexity.VERY_COMPLEX
    
    @classmethod
    def detect_complexity_from_text(cls, text: str) -> ContentComplexity:
        """
        Detect complexity from text content
        
        Args:
            text: Text content to analyze
            
        Returns:
            Estimated complexity
        """
        indicators = {}
        
        # Check for tables (simplified heuristic)
        indicators["has_tables"] = any(
            marker in text.lower() 
            for marker in ["table", "|", "column", "row"]
        )
        
        # Check for diagrams/figures
        indicators["has_diagrams"] = any(
            marker in text.lower()
            for marker in ["figure", "diagram", "chart", "graph", "image"]
        )
        
        # Check for code
        indicators["has_code"] = any(
            marker in text
            for marker in ["def ", "class ", "function", "import ", "```"]
        )
        
        # Estimate text density (words per character)
        words = len(text.split())
        chars = len(text)
        indicators["text_density"] = words / chars if chars > 0 else 0.5
        
        # Count sections (simplified)
        indicators["num_sections"] = text.count("\n\n") + text.count("##")
        
        # Count technical terms (very simplified)
        technical_markers = ["system", "process", "method", "algorithm", "data", 
                            "analysis", "implementation", "framework", "architecture"]
        indicators["technical_terms"] = sum(
            text.lower().count(term) for term in technical_markers
        )
        
        return cls.detect_complexity(indicators)
    
    @classmethod
    def get_budget_recommendation(
        cls,
        scenario: str,
        **kwargs
    ) -> Dict[str, any]:
        """
        Get complete budget recommendation for a scenario
        
        Args:
            scenario: Type of operation (vision, text, page_selection)
            **kwargs: Scenario-specific parameters
            
        Returns:
            Dictionary with budget and reasoning
        """
        if scenario == "vision":
            budget = cls.calculate_vision_budget(
                num_pages=kwargs.get("num_pages", 1),
                complexity=kwargs.get("complexity", ContentComplexity.MODERATE),
                priority=kwargs.get("priority", TaskPriority.NORMAL),
                has_tables=kwargs.get("has_tables", False),
                has_diagrams=kwargs.get("has_diagrams", False)
            )
            reasoning = f"Vision analysis: {kwargs.get('num_pages', 1)} pages, " \
                       f"{kwargs.get('complexity', ContentComplexity.MODERATE).value} complexity"
        
        elif scenario == "text":
            budget = cls.calculate_text_budget(
                text_length=kwargs.get("text_length", 1000),
                complexity=kwargs.get("complexity", ContentComplexity.MODERATE),
                priority=kwargs.get("priority", TaskPriority.NORMAL)
            )
            reasoning = f"Text analysis: {kwargs.get('text_length', 1000)} chars, " \
                       f"{kwargs.get('complexity', ContentComplexity.MODERATE).value} complexity"
        
        elif scenario == "page_selection":
            budget = cls.calculate_page_selection_budget(
                num_available_pages=kwargs.get("num_available_pages", 10),
                summary_length=kwargs.get("summary_length", 5000),
                complexity=kwargs.get("complexity", ContentComplexity.MODERATE)
            )
            reasoning = f"Page selection: {kwargs.get('num_available_pages', 10)} pages available"
        
        else:
            budget = 1500
            reasoning = "Default budget"
        
        return {
            "budget": budget,
            "reasoning": reasoning,
            "scenario": scenario
        }
