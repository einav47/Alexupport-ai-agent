"""
Main Alexupport agent
This module contains the main agent which orchestrates the other microagents
"""

from typing import List

from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

from agent.answer_generator import AnswerGenerator
from agent.followup_generator import FollowUpGenerator
from agent.input_refiner import InputRefiner
from agent.is_answerable_agent import IsAnswerableAgent
from agent.is_relevant_agent import IsRelevantAgent
from agent.information_retriever import InformationRetriever, client
from agent.llm_client import LLMClient

LLM_CLIENT = LLMClient()

class AlexupportAgent:
    """Main Alexupport agent class"""

    def __init__(self):

        self.memory = ConversationBufferMemory(return_messages=True)

        # Initializing all microagents
        self.input_refiner = InputRefiner(llm_client=LLM_CLIENT)
        self.information_retriever = InformationRetriever(qdrant_client=client, llm_client=LLM_CLIENT)
        self.is_answerable_agent = IsAnswerableAgent(llm_client=LLM_CLIENT)
        self.answer_generator = AnswerGenerator(llm_client=LLM_CLIENT, chat_history=self.memory)
        self.followup_generator = FollowUpGenerator(llm_client=LLM_CLIENT)
        self.is_relevant_generator = IsRelevantAgent(llm_client=LLM_CLIENT)

    def intro(self) -> str:
        return ("Hi! I’m **Alexupport** 🤖 — your Amazon product assistant.\n\n"
                "Pick a product on the left, then ask me anything about it! I will do my best to help 😊")

    def format_final_answer(self, answer: str, follow_ups: List[str]) -> str:
        """
        Formats the final answer with follow-up questions.

        Parameters:
        - answer: str; The generated answer to the user's query.
        - follow_ups: List[str]; A list of follow-up questions related to the user's query.

        Returns:
        - str; The formatted final answer with follow-up questions.
        """

        formatted_answer = f"""
        {answer}
        Here are some follow-up questions you might consider:
        {"; ".join([f for f in follow_ups])}
        """

        return formatted_answer

    def answer(self, user_query: str, asin: str) -> str:
        try:
            # Step 0 - Append the user query
            self.memory.chat_memory.messages.append(HumanMessage(content=user_query))

            # Step 1 - Refine user query
            refined_query = self.input_refiner.refine_input(user_input=user_query)

            # Step 2 - Retrieve relevant information from the DB
            retrieved_info = self.information_retriever.retrieve_information(
                query=refined_query,
                product_id=asin
            )

            # Step 3 - Check if the query is answerable
            answerable_result = self.is_answerable_agent.check_answerability(
                user_question=refined_query,
                retrieved_info=retrieved_info
            )
            print(f"DEBUG: check_answerability result: {answerable_result}")  # Debug log

            # Ensure the result is unpacked correctly
            if not isinstance(answerable_result, tuple) or len(answerable_result) != 2:
                raise ValueError("check_answerability must return a tuple (bool, str).")

            answerable, reason1 = answerable_result

            if not answerable:
                msg = f"Sorry — I couldn't find information reliable enough to answer that.\n\nReason: {reason1}"

                followup_questions = self.followup_generator.generate_follow_ups(
                    user_question=refined_query,
                    context=retrieved_info
                )

                final_answer = self.format_final_answer(msg, followup_questions)
                self.memory.chat_memory.messages.append(AIMessage(content=final_answer))
                return final_answer

            iteration = 0

            while iteration < 5:
                # Step 4 - Generate answer
                answer = self.answer_generator.generate_answer(
                    user_question=refined_query,
                    context=retrieved_info
                )

                # Step 5 - Check answer relevance
                relevance_result = self.is_relevant_generator.assess_relevance(
                    user_question=refined_query,
                    generated_response=answer,
                    context=retrieved_info
                )
                print(f"DEBUG: assess_relevance result: {relevance_result}")  # Debug log

                # Ensure the result is unpacked correctly
                if not isinstance(relevance_result, tuple) or len(relevance_result) != 2:
                    raise ValueError("assess_relevance must return a tuple (bool, str).")

                relevant, reason2 = relevance_result

                if relevant:
                    # Step 6 - Generate follow-up questions
                    followup_questions = self.followup_generator.generate_follow_ups(
                        user_question=refined_query,
                        context=retrieved_info
                    )

                    final_answer = self.format_final_answer(
                        answer=answer,
                        follow_ups=followup_questions
                    )

                    self.memory.chat_memory.messages.append(AIMessage(content=final_answer))
                    return final_answer

                iteration += 1

            msg = f"Sorry — I couldn't find a relevant answer to that.\n\nReason: {reason2}"

            followup_questions = self.followup_generator.generate_follow_ups(
                user_question=refined_query,
                context=retrieved_info
            )

            final_answer = self.format_final_answer(msg, followup_questions)
            self.memory.chat_memory.messages.append(AIMessage(content=final_answer))
            return final_answer

        except Exception as e:
            error_message = f"An error occurred while processing your request: {e}"
            self.memory.chat_memory.messages.append(AIMessage(content=error_message))
            return error_message
