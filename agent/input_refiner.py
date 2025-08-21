"""
This module defines the InputRefiner microagent,
which is responsible for refining and enriching user questions
"""

from agent.llm_client import LLMClient
from utils.utils import clean_string

class InputRefiner:
    """Microagent class for refining and enriching user questions"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.system_prompt =  """
        You are a question refinement specialist for Amazon product support.

        Given a question from the user, your task is to:
        1. Fix typos and grammatical errors in the user question
        2. Expand abbreviations and clarify ambiguous terms
        3. Maintain the original intent while making the question more specific

        Return only the refined question without any explanations or additional text.
        """

    def refine_input(self, user_input: str) -> str:
        """
        Refines the user input to improve LLM responses.

        Parameters:
        - user_input: str; The original user question to refine.

        Returns:
        - str; The refined user question.
        """

        complete_human_input = clean_string(f"""
        Refine the following user question to improve clarity and specificity.
        Focus on making the question more detailed and relevant to product materials and durability.

        User Question: {user_input}
        """)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": complete_human_input}
        ]

        response = self.llm_client.generate_response(messages=messages)
        return response.strip()
