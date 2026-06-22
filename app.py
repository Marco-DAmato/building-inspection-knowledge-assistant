import os
from pathlib import Path

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"
COLLECTION_NAME = "inspection_reports"


st.set_page_config(
    page_title="Building Inspection Knowledge Assistant",
    layout="wide"
)

st.title("Building Inspection Knowledge Assistant")
st.write(
    "A small RAG-based MVP that helps building inspectors query historical "
    "inspection reports and retrieve relevant defect cases."
)

api_key = st.text_input("OpenAI API Key", type="password")


# @st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# @st.cache_resource
def build_collection():
    model = load_embedding_model()
    client = chromadb.Client()
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        documents = []
        metadatas = []
        ids = []

        for i, path in enumerate(sorted(REPORT_DIR.glob("*.txt"))):
            text = path.read_text(encoding="utf-8")
            documents.append(text)
            metadatas.append({"source": path.name})
            ids.append(f"report_{i}")

        embeddings = model.encode(documents).tolist()

        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    return collection


def search_reports(question, n_results=3):
    model = load_embedding_model()
    collection = build_collection()

    query_embedding = model.encode([question]).tolist()[0]

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )


def generate_answer(question, results, api_key):
    context = "\n\n".join(results["documents"][0])
    sources = [m["source"] for m in results["metadatas"][0]]

    prompt = f"""
You are a building inspection assistant.

Use only the information in the inspection report extracts below.
Do not invent facts outside the provided context.

CONTEXT:
{context}

SOURCE REPORTS:
{sources}

QUESTION:
{question}

Provide:
1. Summary answer
2. Likely causes
3. Recommended actions
4. Relevant source reports. Use only names from SOURCE REPORTS.
"""

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


question = st.text_input(
    "Ask a building inspection question",
    "What are the likely causes of water infiltration in underground parking?"
)

if st.button("Search inspection knowledge base"):
    if not api_key:
        st.warning("Please enter an OpenAI API key.")
    else:
        with st.spinner("Searching reports and generating answer..."):
            results = search_reports(question)
            answer = generate_answer(question, results, api_key)

        st.subheader("AI-generated answer")
        st.write(answer)

        st.subheader("Retrieved source reports")
        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            with st.expander(f"{meta['source']} | distance: {distance:.3f}"):
                st.write(doc)
