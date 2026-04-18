import os
import gc
import tempfile
import time
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import torch
import fitz  # PyMuPDF
from PIL import Image
import io
import pytesseract

# ========================== POPPLER & PDF2IMAGE SETUP ==========================
import pdf2image
from pdf2image.exceptions import PDFInfoNotInstalledError

# Create a safe temp directory for pdf2image if needed on Windows
if "safe_poppler_temp" not in st.session_state:
    st.session_state.safe_poppler_temp = tempfile.mkdtemp()

original_convert = pdf2image.convert_from_path

def safe_convert_from_path(*args, **kwargs):
    kwargs["output_folder"] = st.session_state.safe_poppler_temp
    return original_convert(*args, **kwargs)

pdf2image.convert_from_path = safe_convert_from_path
# ==============================================================================

import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")

# ========================== CONFIG ==========================
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


# ====================== STREAMLIT UI ======================
st.set_page_config(page_title="📄 Multi-Modal RAG", layout="wide")
st.title(" Multi-Modal RAG ")

# ====================== MEMORY MANAGER ======================
def clear_memory():
    """Clear RAM and VRAM"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def reset_all_documents():
    """Reset everything"""
    if st.session_state.doc_cache:
        for doc in st.session_state.doc_cache.values():
            if doc is not None:
                try:
                    doc.close()
                except:
                    pass
        st.session_state.doc_cache = {}
    
    if st.session_state.pdf_files:
        for pdf_info in st.session_state.pdf_files:
            pdf_path = pdf_info.get("path")
            if pdf_path and os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                except:
                    pass
    
    st.session_state.pdf_files = []
    st.session_state.indexed = False
    st.session_state.faiss_index = None
    st.session_state.chunk_mapping = [] # Renamed from page_mapping to reflect chunking
    clear_memory()

def remove_pdf(file_id):
    """Remove specific PDF"""
    if file_id in st.session_state.doc_cache:
        try:
            st.session_state.doc_cache[file_id].close()
        except:
            pass
        del st.session_state.doc_cache[file_id]
    
    pdf_to_remove = None
    for pdf_info in st.session_state.pdf_files:
        if pdf_info["id"] == file_id:
            pdf_to_remove = pdf_info
            break
    
    if pdf_to_remove:
        pdf_path = pdf_to_remove.get("path")
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except:
                pass
        st.session_state.pdf_files.remove(pdf_to_remove)
    
    st.session_state.indexed = False
    st.session_state.faiss_index = None
    st.session_state.chunk_mapping = []
    clear_memory()

# ====================== LOAD MODELS ======================
@st.cache_resource(show_spinner="Loading semantic embedding model...")
def load_embedding_model():
    try:
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None

@st.cache_resource(show_spinner="Loading Qwen2.5-0.5B...")
def load_llm():
    try:
        clear_memory()
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load LLM: {e}")
        return None, None

embedding_model = load_embedding_model()
tokenizer, llm = load_llm()
if embedding_model is None or tokenizer is None or llm is None:
    st.stop()

# ====================== SEMANTIC EMBEDDING ======================
def embed_text_chunk(text):
    if len(text.strip()) == 0:
        text = "[EMPTY CHUNK]"
    return embedding_model.encode(text, convert_to_numpy=True).astype('float32')

def embed_query(text):
    return embedding_model.encode(text, convert_to_numpy=True).astype('float32')

# ====================== SESSION STATE ======================
for key in ["pdf_files", "indexed", "doc_cache", "faiss_index", "chunk_mapping"]:
    if key not in st.session_state:
        if key in ["pdf_files", "chunk_mapping"]:
            st.session_state[key] = []
        elif key == "doc_cache":
            st.session_state[key] = {}
        else:
            st.session_state[key] = None if key != "indexed" else False

# ====================== UPLOAD ======================
st.markdown("## Step 1: Upload PDFs")
uploaded_file = st.file_uploader(" Upload PDF", type=["pdf"], accept_multiple_files=False)

if uploaded_file:
    file_exists = any(pdf["name"] == uploaded_file.name for pdf in st.session_state.pdf_files)
    if not file_exists:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_path = tmp.name
            
            doc = fitz.open(temp_path)
            page_count = doc.page_count
            doc.close()
            
            if page_count == 0:
                st.error("PDF has no pages")
                os.remove(temp_path)
            else:
                file_id = f"pdf_{int(time.time() * 1000)}"
                st.session_state.pdf_files.append({
                    "id": file_id,
                    "name": uploaded_file.name,
                    "path": temp_path,
                    "page_count": page_count
                })
                st.session_state.indexed = False
                st.success(f" Added: {uploaded_file.name} ({page_count} pages)")
                st.rerun()
        except Exception as e:
            st.error(f"Failed to upload: {e}")
    else:
        st.info(f" {uploaded_file.name} already uploaded")

# ====================== MANAGE DOCUMENTS ======================
st.markdown("## Uploaded Documents")
if st.session_state.pdf_files:
    for pdf_info in st.session_state.pdf_files:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"**{pdf_info['name']}**")
        with col2:
            st.metric("Pages", pdf_info['page_count'])
        with col3:
            if st.button("🗑️", key=f"rm_{pdf_info['id']}"):
                remove_pdf(pdf_info['id'])
                st.rerun()
    if st.button("Clear All", type="secondary"):
        reset_all_documents()
        st.rerun()
else:
    st.info(" Upload PDFs to start")

# ====================== INDEXING ======================
st.markdown("## Step 2: Index Documents (Text, Tables, Images)")

col1, col2 = st.columns([3, 1])
with col1:
    index_button = st.button(
        f" Index {len(st.session_state.pdf_files)} PDF(s)",
        use_container_width=True,
        type="primary",
        disabled=len(st.session_state.pdf_files) == 0
    )
with col2:
    st.metric("Status", " Done" if st.session_state.indexed else "⏳")

if index_button:
    clear_memory()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize LangChain Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    try:
        all_embeddings = []
        chunk_mapping = []
        total_pages = sum(p["page_count"] for p in st.session_state.pdf_files)
        current_page = 0
        poppler_warning_shown = False
        
        for pdf_idx, pdf_info in enumerate(st.session_state.pdf_files):
            status_text.text(f"Processing: {pdf_info['name']}...")
            doc = fitz.open(pdf_info["path"])
            
            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    
                    # 1. Base Text Extraction
                    page_text = page.get_text()
                    
                    # 2. Table Extraction
                    tables = page.find_tables()
                    if tables:
                        page_text += "\n\n[EXTRACTED TABLES]\n"
                        for tab in tables:
                            df = tab.to_pandas()
                            page_text += df.to_markdown(index=False) + "\n\n"
                    
                    # 3. Image OCR Extraction
                    image_list = page.get_images(full=True)
                    if image_list:
                        page_text += "\n\n[EXTRACTED IMAGE TEXT]\n"
                        for img_index, img_info in enumerate(image_list):
                            xref = img_info[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            try:
                                img = Image.open(io.BytesIO(image_bytes))
                                ocr_text = pytesseract.image_to_string(img).strip()
                                if ocr_text:
                                    page_text += f"Image {img_index+1}: {ocr_text}\n"
                            except PDFInfoNotInstalledError:
                                if not poppler_warning_shown:
                                    st.warning("⚠️ Poppler is not installed or not in PATH. OCR for complex PDFs might fail. Please check the README.")
                                    poppler_warning_shown = True
                            except Exception as e:
                                pass # Skip unreadable images
                    
                    # 4. Smart Chunking (LangChain)
                    chunks = text_splitter.split_text(page_text)
                    
                    for chunk_idx, chunk_text in enumerate(chunks):
                        embedding = embed_text_chunk(chunk_text)
                        all_embeddings.append(embedding)
                        
                        chunk_mapping.append({
                            "pdf_id": pdf_info["id"],
                            "pdf_name": pdf_info["name"],
                            "page_num": page_num + 1,
                            "chunk_idx": chunk_idx,
                            "extracted_text": chunk_text 
                        })
                    
                    current_page += 1
                    progress_bar.progress(min(current_page / total_pages, 0.99))
                    
                except Exception as e:
                    st.warning(f" Page {page_num + 1}: {str(e)[:50]}")
                    continue
            doc.close()
        
        if not all_embeddings:
            st.error("No pages could be indexed")
            st.stop()
        
        embeddings_array = np.array(all_embeddings).astype('float32')
        faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
        faiss_index.add(embeddings_array)
        
        st.session_state.faiss_index = faiss_index
        st.session_state.chunk_mapping = chunk_mapping
        
        for pdf_info in st.session_state.pdf_files:
            st.session_state.doc_cache[pdf_info["id"]] = fitz.open(pdf_info["path"])
        
        st.session_state.indexed = True
        progress_bar.progress(1.0)
        status_text.text(" Multi-Modal Indexing complete!")
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        st.error(f"Indexing failed: {e}")
        st.session_state.indexed = False

# ====================== QUERY & CITATION ======================
st.markdown("## Step 3: Ask Questions")

if not st.session_state.indexed:
    st.info("Index documents first (Step 2)")
    st.stop()

query = st.text_input("Your question:", placeholder="Summarize the revenue chart on page 5...")

if st.button(" Search", use_container_width=True, type="primary") and query:
    try:
        clear_memory()
        with st.spinner(" Semantic search in progress..."):
            query_embedding = embed_query(query)
            # Retrieve top 4 chunks instead of 3 pages, since chunks are smaller
            k = min(4, len(st.session_state.chunk_mapping)) 
            distances, indices = st.session_state.faiss_index.search(
                np.array([query_embedding]).astype('float32'), k
            )
        
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(st.session_state.chunk_mapping):
                results.append(st.session_state.chunk_mapping[idx])
        
        if not results:
            st.warning("No results found")
            st.stop()
        
        st.markdown("### Retrieved Context")
        context = ""
        
        # Keep track of shown images to avoid duplicating if multiple chunks hit the same page
        shown_pages = set() 
        
        for i, result in enumerate(results, 1):
            pdf_id = result["pdf_id"]
            page_num = result["page_num"] - 1
            pdf_name = result["pdf_name"]
            doc = st.session_state.doc_cache[pdf_id]
            
            augmented_text = result["extracted_text"]
            
            context += f"\n\n<SOURCE id='{pdf_name} - Page {result['page_num']}'>\n{augmented_text}\n</SOURCE>"
            
            st.write(f"**#{i}. 📄 {pdf_name}** - Page {result['page_num']} (Chunk {result['chunk_idx']})")
            
            page_identifier = f"{pdf_id}_{page_num}"
            if page_identifier not in shown_pages:
                try:
                    pix = doc[page_num].get_pixmap(dpi=72)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    st.image(img, use_container_width=True)
                    shown_pages.add(page_identifier)
                except:
                    st.caption("(Could not render image)")
            st.divider()
        
        # Generate Answer
        st.markdown("### Answer")
        with st.spinner(" Generating precise answer with citations..."):
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert analytical assistant. Answer the user's question based strictly on the provided <SOURCE> contexts. 
Rules:
1. Include information from tables or images if provided in the text.
2. SOURCE ATTRIBUTION: Every time you state a fact or pull data, you MUST cite the source inline at the end of the sentence using the format `[Document Name - Page X]`.
3. Do not invent information. If the answer is not in the context, say "I cannot find this in the uploaded documents." """
                },
                {
                    "role": "user",
                    "content": f"Context data:\n{context}\n\nQuestion: {query}"
                }
            ]
            
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            
            with torch.no_grad():
                outputs = llm.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            input_len = inputs['input_ids'].shape[1]
            answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        
        st.info(answer)
        
        with st.expander("Retrieval Details"):
            for i, (result, dist) in enumerate(zip(results, distances[0]), 1):
                similarity = 1 - (dist / 2)
                st.write(f"{i}. {result['pdf_name']} (Page {result['page_num']}, Chunk {result['chunk_idx']}) - Similarity: {similarity:.2%}")
                
    except Exception as e:
        st.error(f"Error: {e}")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("### System Info")
    st.write(f" PDFs: {len(st.session_state.pdf_files)}")
    st.write(f" Chunks Indexed: {len(st.session_state.chunk_mapping)}")
    st.write(f" Search Model: {EMBEDDING_MODEL}")
    st.write(f" LLM: {LLM_MODEL}")
    st.write(f"Status: {' Ready' if st.session_state.indexed else ' Not indexed'}")
    
    st.divider()
    if st.button(" Clear Memory"):
        clear_memory() 
        st.success("Memory cleared!")
    if st.button(" Reset All"):
        reset_all_documents()
        st.rerun()