"""
Simplified Grid Descriptor for Combined Images
Replaces complex grid descriptions with vision-optimized format
"""

from typing import List, Tuple, Optional


class SimplifiedGridDescriptor:
    """
    Creates simple, vision-optimized descriptions of grid layouts
    Reduces token usage and improves GPT-5 Vision understanding
    """
    
    @staticmethod
    def describe_layout(num_pages: int, grid_size: Tuple[int, int]) -> str:
        """
        Create simplified grid description optimized for vision models
        
        Args:
            num_pages: Total number of pages
            grid_size: (rows, cols) grid dimensions
            
        Returns:
            Simple, clear grid description
        """
        rows, cols = grid_size
        
        if num_pages == 1:
            return "📄 Single page document"
        
        if num_pages <= 4:
            # For small grids, use simple left-to-right description
            positions = SimplifiedGridDescriptor._get_positions_simple(num_pages, rows, cols)
            return f"📄 {num_pages} pages arranged: {', '.join(positions)}"
        
        # For larger grids, use compact description
        return f"📄 {num_pages} pages in {rows}×{cols} grid (left→right, top→bottom)"
    
    @staticmethod
    def _get_positions_simple(num_pages: int, rows: int, cols: int) -> List[str]:
        """
        Get simple position descriptions for small grids
        
        Args:
            num_pages: Number of pages
            rows: Grid rows
            cols: Grid columns
            
        Returns:
            List of simple position descriptions
        """
        positions = []
        page_num = 1
        
        position_names = {
            (0, 0): "top-left",
            (0, 1): "top-right",
            (1, 0): "bottom-left",
            (1, 1): "bottom-right",
            (0, 2): "top-right",
            (1, 2): "bottom-right"
        }
        
        for row in range(rows):
            for col in range(cols):
                if page_num > num_pages:
                    break
                
                if (row, col) in position_names:
                    pos = position_names[(row, col)]
                else:
                    pos = f"row {row+1}, col {col+1}"
                
                positions.append(f"P{page_num} ({pos})")
                page_num += 1
        
        return positions
    
    @staticmethod
    def describe_with_labels(num_pages: int, grid_size: Tuple[int, int]) -> str:
        """
        Create description emphasizing page labels (P1, P2, etc.)
        
        Args:
            num_pages: Total number of pages
            grid_size: (rows, cols) grid dimensions
            
        Returns:
            Description focusing on page labels
        """
        if num_pages == 1:
            return "📄 Page labeled: P1"
        
        page_labels = [f"P{i}" for i in range(1, num_pages + 1)]
        rows, cols = grid_size
        
        return f"📄 Pages labeled {', '.join(page_labels)} in {rows}×{cols} grid"
    
    @staticmethod
    def describe_for_qa(
        num_pages: int,
        grid_size: Tuple[int, int],
        relevant_pages: Optional[List[int]] = None
    ) -> str:
        """
        Create description optimized for Q&A tasks
        
        Args:
            num_pages: Total number of pages
            grid_size: (rows, cols) grid dimensions
            relevant_pages: List of page numbers that may be relevant
            
        Returns:
            Q&A-optimized description
        """
        if num_pages == 1:
            return "📄 Analyzing 1 page (labeled P1)"
        
        rows, cols = grid_size
        base = f"📄 Analyzing {num_pages} pages (labeled P1-P{num_pages}) in {rows}×{cols} grid"
        
        if relevant_pages:
            relevant_str = ', '.join([f"P{p}" for p in relevant_pages])
            base += f"\n🔍 Focus on: {relevant_str}"
        
        return base
    
    @staticmethod
    def describe_for_summary(num_pages: int, grid_size: Tuple[int, int]) -> str:
        """
        Create description optimized for summary generation
        
        Args:
            num_pages: Total number of pages
            grid_size: (rows, cols) grid dimensions
            
        Returns:
            Summary-optimized description
        """
        if num_pages == 1:
            return "📄 Summarizing 1 page"
        
        rows, cols = grid_size
        return f"📄 Summarizing {num_pages} pages (arranged {rows}×{cols}, read left→right, top→bottom)"
    
    @staticmethod
    def get_page_reference_instruction() -> str:
        """
        Get instruction for how to reference pages in responses
        
        Returns:
            Page reference instruction
        """
        return """
📍 When referencing content, use page labels (P1, P2, etc.) visible in the image.
Example: "According to P3..." or "As shown in P1 and P2..."
"""
    
    @staticmethod
    def create_complete_context(
        num_pages: int,
        grid_size: Tuple[int, int],
        task_type: str = "analysis",
        relevant_pages: Optional[List[int]] = None
    ) -> str:
        """
        Create complete context description for combined image
        
        Args:
            num_pages: Total number of pages
            grid_size: (rows, cols) grid dimensions
            task_type: Type of task (analysis, qa, summary)
            relevant_pages: List of relevant page numbers
            
        Returns:
            Complete context description
        """
        if task_type == "qa":
            layout = SimplifiedGridDescriptor.describe_for_qa(
                num_pages, grid_size, relevant_pages
            )
        elif task_type == "summary":
            layout = SimplifiedGridDescriptor.describe_for_summary(num_pages, grid_size)
        else:
            layout = SimplifiedGridDescriptor.describe_layout(num_pages, grid_size)
        
        reference_instruction = SimplifiedGridDescriptor.get_page_reference_instruction()
        
        return f"{layout}\n{reference_instruction}"


class GridLayoutCalculator:
    """
    Helper to calculate optimal grid layouts
    """
    
    @staticmethod
    def calculate_grid_size(num_pages: int, max_cols: int = 3) -> Tuple[int, int]:
        """
        Calculate optimal grid dimensions for number of pages
        
        Args:
            num_pages: Number of pages to arrange
            max_cols: Maximum columns (default 3 for 3x3 max)
            
        Returns:
            (rows, cols) tuple
        """
        if num_pages == 1:
            return (1, 1)
        elif num_pages == 2:
            return (1, 2)
        elif num_pages <= 4:
            return (2, 2)
        elif num_pages <= 6:
            return (2, 3)
        else:
            # For 7-9 pages, use 3x3
            return (3, 3)
    
    @staticmethod
    def get_position(page_index: int, grid_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Get (row, col) position for page in grid
        
        Args:
            page_index: 0-based page index
            grid_size: (rows, cols) tuple
            
        Returns:
            (row, col) position
        """
        rows, cols = grid_size
        row = page_index // cols
        col = page_index % cols
        return (row, col)
