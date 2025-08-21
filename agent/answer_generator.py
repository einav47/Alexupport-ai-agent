"""
This module defines the AnswerGenerator microagent,
which is responsible for generating answers to user questions based on the retrieved information.
"""

from typing import List

from langchain.memory import ConversationBufferMemory

from agent.llm_client import LLMClient
from utils.utils import clean_string

class AnswerGenerator:
    """Microagent class for generating answers to user questions"""

    def __init__(self, llm_client: LLMClient, chat_history: ConversationBufferMemory):
        self.llm_client = llm_client
        self.chat_history = chat_history
        self.system_prompt = """
        You are Alexupport, an expert Amazon product support assistant.
        Your role is to provide helpful, accurate answers based on real customer experiences and verified information.

        Guidelines:
        1. Base your answers ONLY on the provided information from customer reviews and Q&A
        2. Be specific and detailed in your responses
        3. Mention specific product features, experiences, or issues when relevant
        4. Use a friendly, professional tone
        5. If there are conflicting opinions, acknowledge different perspectives
        6. Don't speculate or make claims not supported by the data
        7. Keep responses concise but informative

        Format your response as a clear, helpful answer that directly addresses the user's question.
        """

    def generate_answer(self, user_question: str, context: List[List[str]]) -> str:
        """
        Generates an answer to the user question based on the provided context.

        Parameters:
        - user_question: str; The question posed by the user.
        - context: List[str]; The context information retrieved for the product.

        Returns:
        - str; The generated answer to the user question.
        """

        # Building the message for the LLM

        if self.chat_history:
            history_string = "; ".join(f"{message.type.upper()}: {message.content}" for message in self.chat_history.chat_memory.messages)
            history_prefix = f"Here's the history of the current chat: [{history_string}]."
        else:
            history_prefix = ""

        complete_human_input = clean_string(f"""
        {history_prefix}
        Based on the following information from real customer experiences and reviews, answer the following question using the available information.

        Question: {user_question}

        Available Information:
        {context}

        Provide a helpful, accurate answer based on this information.
        """)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "human", "content": complete_human_input}
        ]

        answer = self.llm_client.generate_response(messages=messages)
        return answer.strip()
