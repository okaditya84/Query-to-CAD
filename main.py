import os
import base64
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Any
from io import BytesIO

import streamlit as st
import numpy as np
from PIL import Image
import pdfplumber
from pdf2image import convert_from_path

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_EXTS = [".pdf"]
VECTOR_STORE_DIR = "cad_vector_store"
SCAD_TEMPLATE = """
module {name}({params}) {{
    {code}
}}
"""

class CADVisionProcessor:
    """Process CAD PDFs using vision models to extract design parameters"""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract page images from PDF using pdf2image"""
        images = []
        try:
            pil_images = convert_from_path(pdf_path)
            for page_num, pil_image in enumerate(pil_images):
                img_byte_arr = BytesIO()
                pil_image.save(img_byte_arr, format="JPEG")
                image_bytes = img_byte_arr.getvalue()
                images.append({
                    "page": page_num + 1,
                    "index": 0,
                    "bytes": image_bytes,
                    "format": "jpeg"
                })
        except Exception as e:
            logger.error(f"Error extracting images: {str(e)}")
        return images
    
    def analyze_image_with_groq(self, image_bytes: bytes) -> str:
        """Analyze CAD image using Groq's vision model"""
        try:
            from groq import Groq
            groq_api_key = os.environ.get("GROQ_API_KEY")
            
            if not groq_api_key:
                return '{"error": "GROQ_API_KEY not found in environment variables"}'
            
            client = Groq(api_key=groq_api_key)
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            response = client.chat.completions.create(
                model="llama-3.2-90b-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this CAD diagram and extract:\n"
                                "1. All dimensional parameters with units\n"
                                "2. Geometric shapes and their relationships\n"
                                "3. Manufacturing specifications\n"
                                "4. Material properties\n"
                                "5. Any annotations or symbols\n"
                                "Return as structured JSON"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }],
                temperature=0.1,
                max_tokens=2048
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            return f'{{"error": "Analysis failed: {str(e)}"}}'
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process PDF document with text and image analysis"""
        documents = []
        
        # Process images
        images = self.extract_images_from_pdf(pdf_path)
        for img in images:
            try:
                analysis = self.analyze_image_with_groq(img["bytes"])
                doc = Document(
                    page_content=f"CAD Image Analysis:\n{analysis}",
                    metadata={
                        "source": pdf_path,
                        "page": img["page"],
                        "type": "image_analysis"
                    }
                )
                documents.append(doc)
            except Exception as e:
                logger.error(f"Image processing error: {str(e)}")
        
        # Process text content
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                if text.strip():
                    text_docs = self.text_splitter.split_documents([
                        Document(page_content=text, metadata={"source": pdf_path, "type": "text_content"})
                    ])
                    documents.extend(text_docs)
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
        
        return documents

class SCADGenerator:
    """Generate OpenSCAD code using RAG and LLMs"""
    
    def __init__(self):
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a CAD engineering assistant. Generate valid OpenSCAD code based on user requirements.\n"
                "Available parameters:\n{context}\n\n"
                "Requirements:\n"
                "1. Use metric units (mm)\n"
                "2. Follow OpenSCAD best practices\n"
                "3. Include parameters for customization\n"
                "4. Add comments explaining key sections\n"
                "5. Use {name} as the module name\n"
                "6. Return only the SCAD code without markdown"
            )),
            ("user", "{query}")
        ])
        
        self.llm = self._initialize_llm()
        self.embeddings = self._initialize_embeddings()
        
    def _initialize_llm(self):
        """Initialize Groq LLM with fallback"""
        try:
            return ChatGroq(
                temperature=0.3,
                model_name="mixtral-8x7b-32768",
                api_key=os.getenv("GROQ_API_KEY")
            )
        except Exception as e:
            logger.error(f"LLM initialization failed: {str(e)}")
            return None
            
    def _initialize_embeddings(self):
        """Initialize embeddings with fallback"""
        try:
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                api_key=os.getenv("GOOGLE_API_KEY")
            )
        except Exception as e:
            logger.error(f"Embeddings initialization failed: {str(e)}")
            return None
    
    def retrieve_designs(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve similar designs from vector store"""
        if not self.embeddings:
            return []
            
        try:
            return Chroma(
                persist_directory=VECTOR_STORE_DIR,
                embedding_function=self.embeddings,
                collection_name="cad_designs",
                client_settings=ChromaSettings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=VECTOR_STORE_DIR
                )
            ).similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Vector store retrieval failed: {str(e)}")
            return []
    
    def generate_code(self, query: str, name: str = "GeneratedDesign") -> str:
        """Generate SCAD code using RAG pipeline"""
        try:
            context_docs = self.retrieve_designs(query)
            context = "\n\n".join([doc.page_content for doc in context_docs])
            
            if self.llm:
                chain = self.prompt_template | self.llm | StrOutputParser()
                return chain.invoke({
                    "context": context,
                    "query": query,
                    "name": name
                })
            return f"module {name}() {{\n    // Error: LLM not available\n}}"
        except Exception as e:
            logger.error(f"Code generation failed: {str(e)}")
            return f"module {name}() {{\n    // Error: {str(e)}\n}}"

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    defaults = {
        "vector_store": None,
        "processed_files": set(),
        "scad_code": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def save_to_vector_store(docs: List[Document]):
    """Create or update Chroma vector store"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        if os.path.exists(VECTOR_STORE_DIR):
            vector_store = Chroma(
                persist_directory=VECTOR_STORE_DIR,
                embedding_function=embeddings,
                collection_name="cad_designs",
                client_settings=ChromaSettings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=VECTOR_STORE_DIR
                )
            )
            vector_store.add_documents(docs)
        else:
            vector_store = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=VECTOR_STORE_DIR,
                collection_name="cad_designs",
                client_settings=ChromaSettings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=VECTOR_STORE_DIR
                )
            )
            
        vector_store.persist()
        return vector_store
    except Exception as e:
        logger.error(f"Vector store save failed: {str(e)}")
        return None

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="CAD Design Generator",
        page_icon="🖨️",
        layout="wide"
    )
    
    initialize_session_state()
    processor = CADVisionProcessor()
    scad_gen = SCADGenerator()
    
    st.title("AI-Powered CAD Design Generator")
    
    with st.sidebar:
        st.header("Training Data Management")
        uploaded_files = st.file_uploader(
            "Upload CAD PDFs",
            type=SUPPORTED_EXTS,
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Process Documents"):
            with st.spinner("Analyzing CAD documents..."):
                try:
                    new_docs = []
                    for file in uploaded_files:
                        if file.name in st.session_state.processed_files:
                            continue
                            
                        with tempfile.NamedTemporaryFile(delete=False) as tmp:
                            tmp.write(file.getvalue())
                            docs = processor.process_pdf(tmp.name)
                            new_docs.extend(docs)
                            
                        st.session_state.processed_files.add(file.name)
                    
                    if new_docs:
                        save_to_vector_store(new_docs)
                        st.success(f"Processed {len(new_docs)} document chunks!")
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")
        
        st.divider()
        st.markdown("### System Status")
        if os.path.exists(VECTOR_STORE_DIR):
            st.success("Vector store ready")
        else:
            st.info("Upload PDFs to initialize")
    
    tab1, tab2 = st.tabs(["Design Generator", "Knowledge Base"])
    
    with tab1:
        st.header("Generate New Design")
        
        col1, col2 = st.columns([3, 2])
        with col1:
            design_query = st.text_area(
                "Design Requirements:",
                height=200,
                placeholder="Create a hemisphere with outer radius 10mm and inner radius 5mm"
            )
            design_name = st.text_input("Module Name:", value="CustomDesign")
            
            if st.button("Generate SCAD Code"):
                if design_query.strip():
                    with st.spinner("Engineering your design..."):
                        st.session_state.scad_code = scad_gen.generate_code(design_query, design_name)
                        st.rerun()
                else:
                    st.warning("Please enter design requirements")
        
        with col2:
            if st.session_state.scad_code:
                st.download_button(
                    label="Download SCAD File",
                    data=st.session_state.scad_code,
                    file_name=f"{design_name}.scad",
                    mime="text/x-openscad"
                )
                st.divider()
                st.subheader("Generated Code")
                st.code(st.session_state.scad_code, language="openscad")
    
    with tab2:
        st.header("Knowledge Base Explorer")
        if os.path.exists(VECTOR_STORE_DIR):
            search_query = st.text_input("Search knowledge base:")
            k_results = st.slider("Results to show", 1, 10, 3)
            
            if search_query:
                try:
                    results = Chroma(
                        persist_directory=VECTOR_STORE_DIR,
                        embedding_function=GoogleGenerativeAIEmbeddings(),
                        collection_name="cad_designs"
                    ).similarity_search(search_query, k=k_results)
                    
                    for idx, doc in enumerate(results):
                        with st.expander(f"Result {idx+1} from {Path(doc.metadata['source']).name}"):
                            st.json(doc.metadata)
                            st.text(doc.page_content[:500] + "...")
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
        else:
            st.info("Process PDFs to populate knowledge base")

if __name__ == "__main__":
    main()