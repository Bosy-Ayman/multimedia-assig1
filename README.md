# Multi-Modal Document Intelligence (RAG-Based QA System)

This repository contains a robust, multi-modal Retrieval-Augmented Generation (RAG) system built to parse and extract insights from complex documents. Unlike traditional text-only RAG pipelines, this system intelligently processes **text, tables, and scanned images (via OCR)** to provide highly accurate, citation-backed answers.

##  Features

* **Multi-Modal Ingestion:** Extracts standard text, parses tables into Markdown formats, and runs OCR on embedded images using PyMuPDF and Tesseract.
* **Smart Semantic Chunking:** Utilizes LangChain's `RecursiveCharacterTextSplitter` to segment dense pages intelligently, preserving context without exceeding embedding limits.
* **Unified Vector Indexing:** Embeds all multi-modal data into a unified FAISS vector store using `all-MiniLM-L6-v2`.
* **Strict Source Attribution:** Forces the LLM (Qwen2.5) to cite specific document names, page numbers, and chunk indices for every generated fact to prevent hallucinations.
* **Interactive UI:** A clean, memory-managed Streamlit interface supporting document upload, indexing status, and visual context retrieval.

---

##  System Requirements

To utilize the image extraction and OCR capabilities, you **must** install the following system dependencies alongside the Python packages.

### 1. Tesseract OCR
Required for extracting text from images embedded within PDFs.
* **Windows:** Download the installer from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and add the installation path (e.g., `C:\Program Files\Tesseract-OCR`) to your system's `PATH` environment variable.
* **macOS:** `brew install tesseract`
* **Linux (Ubuntu):** `sudo apt-get install tesseract-ocr`

### 2. Poppler
Required by `pdf2image` for robust PDF-to-image conversion during fallback processing.
* **Windows:** Download the latest binary from [oschwartz10612/poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases) and add the `bin/` folder to your system's `PATH`.
* **macOS:** `brew install poppler`
* **Linux (Ubuntu):** `sudo apt-get install poppler-utils`

---

## Installation & Setup

**1. Clone the repository**
```bash
git clone <Bosy-Ayman/multimedia-assig1>
cd <Bosy-Ayman/multimedia-assig1>
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**3. Install Python dependencies**
```bash
pip install -r requirements.txt
```
*(Ensure `langchain-text-splitters`, `streamlit`, `transformers`, `torch`, `faiss-cpu`, `PyMuPDF`, `pytesseract`, and `pdf2image` are in your `requirements.txt`)*

**4. Run the application**
```bash
streamlit run app.py
```

---

##  Architecture & Tech Stack

* **Frontend Interface:** [Streamlit](https://streamlit.io/)
* **LLM Generator:** `Qwen/Qwen2.5-0.5B-Instruct` (Run locally via HuggingFace Transformers)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Document Processing:** PyMuPDF (`fitz`) for text/tables/images, LangChain for chunking, PyTesseract for OCR.

---

## 📂 Repository Structure

```text
├── app.py                 # Main Streamlit application and RAG pipeline
├── requirements.txt       # Python package dependencies
├── README.md              # Project documentation
└── sample_data/           # Directory containing test PDFs for multi-modal evaluation
``` 

## Video demo
[![▶ Watch Demo](thumbnail.png)](https://drive.google.com/file/d/1Tjqc86HEHlYHbMCuErXh2vZJEjLlwKnk/view?usp=sharing)
