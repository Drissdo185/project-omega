"""
Centralized prompt templates for AI components.
"""

# Document summarization prompt
SUMMARIZATION_PROMPT = """Analyze all the pages in this document and provide a comprehensive summary.

Your summary should:
1. Identify the main topic and purpose of the document
2. List the key sections or themes covered
3. Highlight important facts, figures, or findings
4. Note the document type (e.g., report, article, manual, etc.)
5. Mention any significant details that would help with future queries

Be detailed and thorough. This summary will be used to help answer questions about the document.

Provide the summary in a clear, well-structured format."""


# Page selection prompt
PAGE_SELECTION_PROMPT = """You are helping to find relevant pages in a document to answer a specific query.

Document Summary:
{summary}

Query: {query}

Task: Look at all the pages in this document and identify which pages are most relevant to answering the query.

Instructions:
- Carefully examine each page
- Select pages that contain information directly relevant to the query
- You may select up to {max_pages} pages
- Return ONLY a JSON array of page numbers (1-indexed)
- Order pages by relevance (most relevant first)

Example response format:
[3, 7, 12]

If no pages are relevant, return an empty array: []

Response (JSON array only):"""


# Page analysis prompt
ANALYSIS_PROMPT = """Answer the following query based on the content shown in these pages.

Query: {query}

Instructions:
- Read all the pages carefully
- Provide a direct answer to the query based on what you see
- Quote or reference specific information from the pages when relevant
- If the information is not sufficient to answer the query, say so
- Be concise but thorough

Your response:"""


# Response synthesis prompt
SYNTHESIS_PROMPT = """You are synthesizing information from multiple page analyses to provide a final answer.

Query: {query}

Page Analyses:
{analyses}

Task: Combine the findings from the analyses above into a coherent, comprehensive final answer.

Instructions:
- Integrate information from all analyses
- Resolve any contradictions or duplications
- Organize the information logically
- Provide a clear, well-structured answer
- If the analyses don't fully answer the query, note what's missing

Your final synthesized answer:"""


# Task planning prompt (for RAGAgent)
TASK_PLANNING_PROMPT = """Analyze the following query to determine the best approach for answering it.

Query: {query}

Document Summary:
{summary}

Task: Determine:
1. What type of information is needed to answer this query?
2. What pages or sections might be most relevant?
3. Any special considerations for processing this query?

Provide a brief analysis (2-3 sentences):"""
