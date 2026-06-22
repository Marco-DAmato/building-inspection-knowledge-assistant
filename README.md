# Building Inspection Knowledge Assistant

## Overview
A lightweight Retrieval-Augmented Generation (RAG) assistant for building inspectors and technical risk assessors. The goal is to make historical inspection knowledge easier to search and reuse.

## Problem
Building inspection knowledge is often stored in historical reports and technical documentation, making it difficult to quickly identify similar cases, likely causes, and recommended actions.

## Solution
This project combines semantic search and a Large Language Model to retrieve relevant inspection reports and generate concise, evidence-based recommendations.

The workflow is:

1. Inspection reports are stored as text documents.
2. Reports are converted into embeddings using Sentence Transformers.
3. Embeddings are indexed in ChromaDB.
4. User questions are matched against the report database.
5. Relevant reports are retrieved.
6. GPT-4o-mini generates a structured answer based on the retrieved reports.

## Relevance to SECO

SECO generates and reviews large volumes of technical inspection reports. This prototype demonstrates how historical inspection knowledge can be transformed into a searchable knowledge base, helping inspectors and risk assessors identify similar cases, understand likely root causes, and accelerate technical decision making.


## Target User
- Building Inspectors
- Technical Risk Assessors
- Structural Engineers

## Tech Stack
- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- OpenAI GPT-4o-mini

The focus was on building a complete and reproducible MVP rather than a production-ready system.

## Example Questions
- What causes water infiltration in underground parking?
- How should facade cracking be investigated?
- What are common causes of foundation settlement?

## Future Improvements

- Support PDF inspection reports
- OCR for scanned documents
- Image analysis of defect photographs
- Risk scoring and defect classification
- Cloud deployment and multi-user support

## Production Considerations

For a production version, I would keep the RAG architecture, semantic retrieval and overall workflow.

I would replace the synthetic reports with real document ingestion pipelines, add authentication, persistent storage and monitoring, and deploy the solution on a cloud platform.

## Run

Create a virtual environment:

python -m venv venv

Install dependencies:

.\venv\Scripts\python.exe -m pip install -r requirements.txt

Launch the application:

.\venv\Scripts\python.exe -m streamlit run app.py

Open the local URL displayed by Streamlit in your browser.

## API Key
An OpenAI API key is required to run the LLM component.

The application will request the API key at runtime through the user interface. The key is not stored in the project.
