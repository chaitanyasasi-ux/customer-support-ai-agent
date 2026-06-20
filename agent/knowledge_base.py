# ============================================================
# agent/knowledge_base.py
#
# Owns everything related to building the RAG knowledge base:
# loading Bitext, cleaning template placeholders, chunking,
# embedding, and building the FAISS index.
# ============================================================

import re
import pandas as pd
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# Bitext dataset uses unfilled template variables like
# {{Customer Support Hours}} — these need realistic replacements
# before the text becomes part of our knowledge base, otherwise
# they leak verbatim into the agent's answers.
PLACEHOLDER_MAP = {
    "{{Customer Support Hours}}":        "9 AM to 6 PM, Monday to Saturday",
    "{{Customer Support Phone Number}}": "1800-XXX-XXXX",
    "{{Customer Support Email}}":        "support@yourcompany.com",
    "{{Delivery Country}}":              "India and select international locations",
    "{{Online Order Interaction}}":      "Order History",
    "{{Online Company Portal Info}}":    "your account page",
    "{{Website URL}}":                   "our website",
    "{{Store Location}}":                "your nearest store",
    "{{Refund Amount}}":                 "the refunded amount",
    "{{Order Number}}":                  "your order number",
}


def clean_placeholders(text: str) -> str:
    """
    Replace known Bitext template variables with realistic defaults.
    Falls back to a generic phrase for any {{...}} pattern we
    didn't explicitly map, so nothing broken ever reaches the user.
    """
    for placeholder, replacement in PLACEHOLDER_MAP.items():
        text = text.replace(placeholder, replacement)
    text = re.sub(r"\{\{.*?\}\}", "our team", text)
    return text


def build_documents() -> list[Document]:
    """
    Load the Bitext customer support dataset and build one clean
    Document per intent (27 total), using the longest/most detailed
    response available for each intent as the knowledge base entry.
    """
    dataset = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train"
    )
    df = pd.DataFrame(dataset)

    best_per_intent = (
        df.assign(resp_len=df["response"].str.len())
          .sort_values("resp_len", ascending=False)
          .groupby("intent")
          .first()
          .reset_index()
    )

    documents = [
        Document(
            page_content=clean_placeholders(row["response"].strip()),
            metadata={"intent": row["intent"], "category": row["category"]}
        )
        for _, row in best_per_intent.iterrows()
    ]
    return documents


def build_vectorstore(documents: list[Document]) -> tuple[FAISS, int]:
    """
    Chunk the documents and build a FAISS index using MiniLM embeddings.
    Returns the vectorstore plus the chunk count.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore, len(chunks)


def build_knowledge_base() -> tuple[FAISS, int, int]:
    """
    Top-level entry point: builds documents, then the vectorstore.
    Returns (vectorstore, num_documents, num_chunks).
    """
    documents = build_documents()
    vectorstore, num_chunks = build_vectorstore(documents)
    return vectorstore, len(documents), num_chunks