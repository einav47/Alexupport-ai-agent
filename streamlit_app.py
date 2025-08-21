"""
Streamlit module for the Alexupport Agent
"""

import time
import os
import streamlit as st
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

# Validate environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")
VECTOR_NAME = os.getenv("QDRANT_VECTOR_NAME")

if not all([QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, VECTOR_NAME]):
    st.error("Missing required environment variables. Please check your .env file.")
    st.stop()

# Initialize Qdrant client
try:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize Qdrant client: {e}")
    st.stop()

from agent.alexupport_agent import AlexupportAgent
from agent.information_retriever import InformationRetriever, client
from agent.llm_client import LLMClient

LLM_CLIENT = LLMClient()

def typing_stream(text):
    """Simulate typing effect"""
    for word in text.split():
        yield word + " "
        time.sleep(0.05)

# Get all points (limit to 2,000 for performance)
points = client.scroll(collection_name=COLLECTION_NAME, limit=2000)[0]

def main():
    st.set_page_config(page_title="Alexupport", page_icon="🤖", layout="wide")
    st.title("🤖 Alexupport — Amazon Product Support Assistant")

    with st.sidebar:
        st.header("Product")
        try:
            retriever = InformationRetriever(
                qdrant_client=client,
                llm_client=LLM_CLIENT
            )
            products = retriever.list_products(limit=400)
        except Exception as e:
            st.error(f"Error retrieving products: {e}")
            st.stop()

        if not products:
            st.info("No products found in the collection.")
            st.stop()


        options = [f"{p['asin']} — {p.get('productTitle', '(untitled)')}" for p in products]
        chosen = st.selectbox("Product", options, index=0)
        selected_asin = chosen.split(" — ", 1)[0]
        product_title = next((p.get("productTitle") for p in products if p["asin"] == selected_asin), None)
        if selected_asin:
            st.caption(f"[Open on Amazon](https://www.amazon.com/dp/{selected_asin})")

    if "agent" not in st.session_state:
        st.session_state["agent"] = AlexupportAgent()
    agent = st.session_state["agent"]

    # ---------- Chat state reset when ASIN changes ----------
    if "asin" not in st.session_state or st.session_state["asin"] != selected_asin:
        st.session_state["asin"] = selected_asin
        st.session_state["messages"] = []

        # clear conversation memory if your agent exposes it
        try:
            agent.memory.clear()  # ok if you use ConversationBufferMemory
        except Exception:
            pass

        # use your agent's intro method
        agent.intro() if hasattr(agent, "intro") else f"Hi! Ask me about {product_title or 'this product'}."


    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": agent.intro()}]

    # ---------- Display chat history ----------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about the product..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                resp = agent.answer(prompt, asin=selected_asin)
            except Exception as e:
                resp = f"Error: {e}"
            st.session_state.messages.append({"role": "assistant", "content": resp})
            st.write_stream(typing_stream(resp))

if __name__ == "__main__":
    main()
