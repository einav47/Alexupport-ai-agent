"""
This module defines the FollowUpGenerator class,
which is responsible for generating follow-up questions based on the user's input and the retrieved information.
"""

from typing import List

from agent.llm_client import LLMClient
from utils.utils import clean_string

class FollowUpGenerator:
    """Microagent class for generating relevant follow-up questions"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.system_prompt = """
        You are a customer experience specialist for Amazon product support.
        Your role is to generate 2-3 relevant follow-up questions that would help users discover more useful information.

        Guidelines:
        1. Questions should be related to the user's original question, and to the retrieved information
        2. Focus on practical concerns users might have
        3. Make questions specific and actionable
        4. Avoid generic questions like "Do you have any other questions?"
        5. Questions should help users make informed decisions

        Format your response as a simple list of 2-3 questions, one per line, without numbering or bullet points.
        """

    def generate_follow_ups(self, user_question: str, context: List[List[str]]) -> List[str]:
        """
        Generates follow-up questions based on the user input and retrieved information.

        Parameters:
        - user_question: str; The question posed by the user.
        - context: List[str]; The context information retrieved for the product.

        Returns:
        - List[str]; A list of generated follow-up questions.
        """

        complete_human_input = clean_string(f"""
        Based on the following conversation, generate 2-3 relevant follow-up questions:

        User question: {user_question}
        Available Information Context:
        {context}

        Generate follow-up questions that would help the user discover more useful information about this product or related concerns.
        Return the follow-up questions as a simple list, each question on a new line, separated by a newline character.
        """)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "human", "content": complete_human_input}
        ]

        follow_ups = self.llm_client.generate_response(messages=messages)

        return [q.strip() for q in follow_ups.split('\n')]
