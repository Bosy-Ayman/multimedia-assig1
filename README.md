# Multi-Modal RAG System - DSAI 413 Assignment 1

Local, private RAG application that handles text, tables, and images from PDFs.

## Features
- Text + Table extraction (PyMuPDF)
- Image OCR (Tesseract) for charts and figures
- Semantic search with all-MiniLM-L6-v2 + FAISS
- Citation-backed answers using Qwen2.5-0.5B-Instruct
- Streamlit UI with page image preview


## How to Run
1. `pip install -r requirements.txt`
2. Install Poppler and Tesseract (Windows paths already handled)
3. `streamlit run app.py`

## Video Demo
