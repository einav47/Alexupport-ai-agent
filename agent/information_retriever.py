"""
This module defines the InformationRetriever microagent,
which is responsible for retrieveing relevant information from the Qdrant database.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PayloadSelector

from agent.llm_client import LLMClient

QDRANT_URL = "https://63ad19dc-7779-4868-bc81-41f5fae4353a.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0._mnughDNgXpg2I_tMDwpIIKZJiDqma2o_YDld0ZseR4"
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

class InformationRetriever:
    """Microagent class for retrieving relevant information from the Qdrant database"""

    def __init__(self, qdrant_client: QdrantClient, llm_client: LLMClient):
        self.qdrant_client = qdrant_client
        self.llm_client = llm_client

    def list_products(self, limit: int = 500) -> list[dict]:
        """Return [{'asin','productTitle'}] found in collection payloads."""
        collected, page_offset, seen = [], None, set()
        while len(collected) < limit:
            resp = self.qdrant_client.scroll(
                collection_name="data_collection",
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=page_offset
            )
            print("DEBUG:", type(resp), hasattr(resp, "points"))

            # Handle both tuple and object responses
            if hasattr(resp, "points"):  # ScrollResult style
                points = resp.points
                page_offset = getattr(resp, "next_page_offset", None)
            else:  # legacy tuple style: (points, next_page_offset) or (points, next_page_offset, has_more)
                points = resp[0]
                page_offset = resp[1] if len(resp) > 1 else None

            if not points:
                break

            for p in points:
                payload = getattr(p, "payload", {}) or {}
                asin = payload.get("asin")
                title = payload.get("productTitle")
                if asin and asin not in seen:
                    seen.add(asin)
                    collected.append({"asin": asin, "productTitle": title or "(untitled)"})

            if page_offset is None:
                break

        return collected

    def retrieve_information(self, query: str, product_id: str) -> list:
        """
        Retrieve relevant information from the Qdrant database based on the user's query.

        Parameters:
        - query (str): The user's question or query.
        - product_id (str): The product ID (ASIN) to filter results.

        Returns:
        - list: A list of relevant documents or snippets related to the query and product ID.
        """

        try:
            query_vector = self.llm_client.generate_embeddings(texts=[query])

            # Use the Qdrant client to perform the search
            search_results = self.qdrant_client.query_points(
                collection_name="data_collection",
                query=query_vector,
                using="questionText",
                limit=10,
                with_payload=True,  # Ensure payloads are included
                query_filter=Filter(
                    must=[
                        FieldCondition(key="asin", match=MatchValue(value=product_id))
                    ]
                )
            )

            # Keep only the points with a similarity score of at least 0.5
            search_results.points = [point for point in search_results.points if point.score >= 0.5]

            # Extract relevant fields from the payloads
            results_answers_and_review_snippets = [
                result.payload.get('answers', []) + result.payload.get('review_snippets', [])
                for result in search_results.points
            ]
            return results_answers_and_review_snippets

        except Exception as e:
            raise RuntimeError(f"Error retrieving information: {e}")
