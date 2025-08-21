"""
This module contains the LLMClient class, which is used to interact with the LLM.
"""

import os
from typing import List, Dict

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import tiktoken

from utils.utils import log_token_usage

tiktoken.encoding_for_model("gpt-4o")

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
if not AZURE_OPENAI_API_KEY:
    raise ValueError("API_KEY is not set, add it to the .env file")

AZURE_OPENAI_ENDPOINT = "https://096290-oai.openai.azure.com"
API_VERSION = "2023-05-15"


class LLMClient:
    """
    This is a general LLM client class to be used across all microagents in the project.
    It initializes the chat and embedding models using AzureOpenAI,
    and provides methods for chat/embedding generation, as well as tokens count.
    """

    def __init__(self):
        """Initializes the LLMClient with Azure OpenAI models"""
        self.chat_model = AzureChatOpenAI(
            azure_deployment="team13-gpt4o",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            openai_api_version=API_VERSION,
            max_tokens=500
        )

        self.embedding_model = AzureOpenAIEmbeddings(
            azure_deployment="team13-embedding",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            openai_api_version=API_VERSION
        )

        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    def count_tokens(self, text: str) -> int:
        """
        Counts the number of tokens in the given text using the tokenizer.

        Parameters:
        - text: str; The text to count tokens for.

        Returns:
        - int; The number of tokens in the text.
        """
        return len(self.tokenizer.encode(text))

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Generates a response from the chat model based on the provided messages.

        Parameters:
        - messages: List[Dict[str, str]]; The messages to send to the chat model.

        Returns:
        - str; The response from the chat model.
        """

        langchain_messages = []
        total_input_tokens = 0

        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "human" or msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
            else:
                langchain_messages.append(HumanMessage(content=msg["content"]))

            total_input_tokens += self.count_tokens(msg["content"])

        response = self.chat_model.invoke(langchain_messages)
        output_tokens = self.count_tokens(response.content)
        log_token_usage(
            operation="response_generation",
            input_tokens=total_input_tokens,
            output_tokens=output_tokens
        )

        return response.content

    def generate_embeddings(self, texts: List[str]) -> List[float]:
        """
        Generates embeddings for the given texts.

        Parameters:
        - texts: List[str]; The texts to generate embeddings for.

        Returns:
        - List[float]; The generated embeddings for the texts.
        """
        total_input_tokens = sum(self.count_tokens(text) for text in texts)

        if len(texts) == 1:
            embeddings = self.embedding_model.embed_query(texts[0])
        else:
            embeddings = self.embedding_model.embed_documents(texts)

        log_token_usage(
            operation="embeddings_generation",
            input_tokens=total_input_tokens
        )

        return embeddings

client = LLMClient()
