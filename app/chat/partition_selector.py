"""
Partition selector that uses partition summaries to identify relevant partitions for large documents.
Only used for documents with >20 pages.
Loads partition data from partition_details.json.
"""

import json
from typing import List, Optional
from loguru import logger

from app.models.document import Document, Partition, PartitionDetails, PartitionDetail
from app.providers.base import BaseProvider
from app.storage.document_store import DocumentStore


class PartitionSelector:
    """Selects relevant partitions based on summaries and user query"""

    def __init__(self, provider: BaseProvider, storage_root: str = None):
        """
        Initialize partition selector

        Args:
            provider: LLM provider for analyzing relevance
            storage_root: Optional storage root for DocumentStore
        """
        self.provider = provider
        self.document_store = DocumentStore(storage_root)

    async def select_relevant_partitions(
        self,
        document: Document,
        user_question: str,
        max_partitions: int = 2,
        model: Optional[str] = None
    ) -> List[Partition]:
        """
        Select most relevant partitions for a user question based on partition summaries.
        Loads partition data from partition_details.json.

        Args:
            document: Document with partition summaries
            user_question: User's question
            max_partitions: Maximum partitions to select (default: 2, range: 1-2)

        Returns:
            List of selected Partition objects, ordered by relevance
        """
        try:
            logger.info(f"Selecting relevant partitions for question: {user_question}")

            # Load partition_details.json
            partition_details = self.document_store.load_partition_details(document.id)

            if not partition_details or not partition_details.partitions:
                logger.warning(f"No partition_details.json found for document {document.id}, falling back to document.partitions")
                if not document.has_partitions():
                    logger.warning("Document has no partitions, returning empty list")
                    return []
                partition_details_list = document.partitions
            else:
                logger.info(f"Loaded {len(partition_details.partitions)} partitions from partition_details.json")
                partition_details_list = partition_details.partitions

            # Build context with all partition summaries
            partitions_context = self._build_partitions_context_from_details(partition_details_list, document.name, document.page_count)

            # Build prompt for partition selection
            prompt = self._build_selection_prompt(user_question, partitions_context, max_partitions)

            # Ask LLM to select relevant partitions
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert document analyst. Identify which partitions of a document are most relevant to answer a user's question."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            response = await self.provider.process_text_messages(
                messages=messages,
                max_tokens=100,
                model=model
            )

            # Parse selected partition IDs
            selected_partition_ids = self._parse_partition_ids(response)

            # Apply max_partitions limit
            if len(selected_partition_ids) > max_partitions:
                logger.info(f"Agent selected {len(selected_partition_ids)} partitions, limiting to {max_partitions}")
                selected_partition_ids = selected_partition_ids[:max_partitions]

            # Get actual Partition objects
            selected_partitions = self._get_partitions_by_ids(partition_details_list, selected_partition_ids)

            # Track cost
            cost = self.provider.get_last_cost() or 0.0
            logger.info(f"Selected {len(selected_partitions)} partitions (cost: ${cost:.4f})")
            logger.debug(f"Selected partition IDs: {selected_partition_ids}")

            return selected_partitions

        except Exception as e:
            logger.error(f"Failed to select relevant partitions: {e}")
            # Fallback: return first partition
            if document.has_partitions():
                return document.partitions[:1]
            return []

    def _build_partitions_context(self, document: Document) -> str:
        """Build formatted context of all partition summaries (legacy method)"""
        lines = [f"Document: {document.name}"]
        lines.append(f"Total pages: {document.page_count}")
        lines.append(f"Total partitions: {len(document.partitions)}")
        lines.append("\nPartition summaries:")

        for partition in document.partitions:
            lines.append(
                f"\nPartition {partition.partition_id} "
                f"(Pages {partition.page_range[0]}-{partition.page_range[1]}): "
                f"{partition.summary}"
            )

        return "\n".join(lines)

    def _build_partitions_context_from_details(
        self,
        partition_details_list: List,
        document_name: str,
        page_count: int
    ) -> str:
        """
        Build formatted context from partition_details.json data.
        Works with both PartitionDetail and Partition objects.
        """
        lines = [f"Document: {document_name}"]
        lines.append(f"Total pages: {page_count}")
        lines.append(f"Total partitions: {len(partition_details_list)}")
        lines.append("\nPartition summaries:")

        for partition in partition_details_list:
            lines.append(
                f"\nPartition {partition.partition_id} "
                f"(Pages {partition.page_range[0]}-{partition.page_range[1]}): "
                f"{partition.summary}"
            )

        return "\n".join(lines)

    def _build_selection_prompt(
        self,
        user_question: str,
        partitions_context: str,
        max_partitions: int
    ) -> str:
        """Build prompt for LLM to select relevant partitions"""

        return f"""Given a user's question and summaries of all partitions in a large document, identify which partitions are most relevant to answer the question.

User Question:
{user_question}

{partitions_context}

Task:
1. Analyze which partitions contain information relevant to the user's question
2. Select up to {max_partitions} most relevant partitions (hard limit)
3. Prioritize quality - only select partitions that will actually help answer the question
4. Return ONLY a JSON array of partition IDs in order of relevance (most relevant first)

Example response format:
[1, 3]

Your response (JSON array only):"""

    def _parse_partition_ids(self, response: str) -> List[int]:
        """Parse partition IDs from LLM response"""
        try:
            # Clean response
            response = response.strip()

            # Try to find JSON array in response
            start = response.find('[')
            end = response.rfind(']') + 1

            if start >= 0 and end > start:
                json_str = response[start:end]
                partition_ids = json.loads(json_str)

                # Validate it's a list of integers
                if isinstance(partition_ids, list):
                    return [int(num) for num in partition_ids if isinstance(num, (int, str)) and str(num).isdigit()]

            logger.warning(f"Could not parse partition IDs from: {response}")
            return []

        except Exception as e:
            logger.error(f"Failed to parse partition IDs: {e}")
            return []

    def _get_partitions_by_ids(
        self,
        partitions_list: List,
        partition_ids: List[int]
    ) -> List[Partition]:
        """
        Get Partition objects by their IDs from a list.
        Converts PartitionDetail to Partition if needed.
        """
        selected_partitions = []

        for partition_id in partition_ids:
            # Find partition with matching ID
            for partition in partitions_list:
                if partition.partition_id == partition_id:
                    # If it's a PartitionDetail, convert to Partition
                    if isinstance(partition, PartitionDetail):
                        selected_partitions.append(Partition(
                            partition_id=partition.partition_id,
                            page_range=partition.page_range,
                            summary=partition.summary
                        ))
                    else:
                        selected_partitions.append(partition)
                    break

        return selected_partitions
