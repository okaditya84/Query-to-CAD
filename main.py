# import os
# import base64
# import tempfile
# import logging
# import shutil  # For checking if poppler is installed
# from pathlib import Path
# from typing import List, Dict, Any
# from io import BytesIO

# import streamlit as st
# import numpy as np
# from PIL import Image
# import pdfplumber
# from pdf2image import convert_from_path

# from langchain_core.documents import Document
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_groq import ChatGroq

# # Import ChromaSettings to configure the vector store client
# from chromadb.config import Settings as ChromaSettings

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Constants
# SUPPORTED_EXTS = [".pdf"]
# VECTOR_STORE_DIR = "cad_vector_store"
# SCAD_TEMPLATE = """
# module {name}({params}) {{
#     {code}
# }}
# """

# class CADVisionProcessor:
#     """Process CAD PDFs using vision models to extract design parameters"""

#     def __init__(self):
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )
        
#     def extract_images_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
#         """Extract page images from PDF using pdf2image and compress if they are too large"""
#         images = []
#         MAX_IMAGE_SIZE = 500 * 1024  # 500 KB threshold
#         MAX_PIXEL_COUNT = 33177600  # Maximum allowed pixels

#         # Check if poppler is installed (pdftoppm must be in PATH)
#         if not shutil.which("pdftoppm"):
#             logger.error("Poppler is not installed or not in PATH. Please install poppler to enable image extraction.")
#             return images  # Return empty list if poppler is not available

#         try:
#             pil_images = convert_from_path(pdf_path)
#             for page_num, pil_image in enumerate(pil_images):
#                 # Check and resize if the image exceeds the allowed pixel count
#                 w, h = pil_image.size
#                 if w * h > MAX_PIXEL_COUNT:
#                     factor = (MAX_PIXEL_COUNT / (w * h)) ** 0.5
#                     new_size = (int(w * factor), int(h * factor))
#                     logger.info(f"Resizing page {page_num + 1} image from {pil_image.size} to {new_size} to meet pixel limit.")
#                     pil_image = pil_image.resize(new_size, Image.ANTIALIAS)

#                 try:
#                     # Save the image to a BytesIO buffer in JPEG format
#                     img_byte_arr = BytesIO()
#                     pil_image.save(img_byte_arr, format="JPEG")
#                     image_bytes = img_byte_arr.getvalue()

#                     # If the image is larger than the threshold, recompress it with lower quality
#                     if len(image_bytes) > MAX_IMAGE_SIZE:
#                         logger.info(f"Page {page_num + 1} image ({len(image_bytes)} bytes) exceeds threshold. Recompressing...")
#                         compressed_arr = BytesIO()
#                         pil_image.save(compressed_arr, format="JPEG", quality=70, optimize=True)
#                         image_bytes = compressed_arr.getvalue()
#                         logger.info(f"Compressed image size: {len(image_bytes)} bytes")
#                 except Exception as inner_e:
#                     logger.error(f"Error processing image for page {page_num + 1}: {str(inner_e)}")
#                     continue

#                 images.append({
#                     "page": page_num + 1,
#                     "index": 0,
#                     "bytes": image_bytes,
#                     "format": "jpeg"
#                 })
#         except Exception as e:
#             logger.error(f"Error extracting images: {str(e)}")
#         return images

    
#     def analyze_image_with_groq(self, image_bytes: bytes) -> str:
#         """Analyze CAD image using Groq's vision model"""
#         try:
#             from groq import Groq
#             groq_api_key = os.environ.get("GROQ_API_KEY")
            
#             if not groq_api_key:
#                 return '{"error": "GROQ_API_KEY not found in environment variables"}'
            
#             client = Groq(api_key=groq_api_key)
#             base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
#             response = client.chat.completions.create(
#                 model="llama-3.2-11b-vision-preview",
#                 messages=[{
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "text",
#                             "text": (
#                                 "Analyze this CAD diagram and extract:\n"
#                                 "1. All dimensional parameters with units\n"
#                                 "2. Geometric shapes and their relationships\n"
#                                 "3. Manufacturing specifications\n"
#                                 "4. Material properties\n"
#                                 "5. Any annotations or symbols\n"
#                                 "Return as structured JSON"
#                             )
#                         },
#                         {
#                             "type": "image_url",
#                             "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
#                         }
#                     ]
#                 }],
#                 temperature=0.5,
#                 max_tokens=5000
#             )
            
#             return response.choices[0].message.content
#         except Exception as e:
#             logger.error(f"Image analysis failed: {str(e)}")
#             return f'{{"error": "Analysis failed: {str(e)}"}}'
    
#     def process_pdf(self, pdf_path: str) -> List[Document]:
#         """Process PDF document with text and image analysis"""
#         documents = []
        
#         # Process images
#         images = self.extract_images_from_pdf(pdf_path)
#         for img in images:
#             try:
#                 analysis = self.analyze_image_with_groq(img["bytes"])
#                 doc = Document(
#                     page_content=f"CAD Image Analysis:\n{analysis}",
#                     metadata={
#                         "source": pdf_path,
#                         "page": img["page"],
#                         "type": "image_analysis"
#                     }
#                 )
#                 documents.append(doc)
#             except Exception as e:
#                 logger.error(f"Image processing error: {str(e)}")
        
#         # Process text content
#         try:
#             with pdfplumber.open(pdf_path) as pdf:
#                 text = "\n".join([page.extract_text() or "" for page in pdf.pages])
#                 if text.strip():
#                     text_docs = self.text_splitter.split_documents([
#                         Document(page_content=text, metadata={"source": pdf_path, "type": "text_content"})
#                     ])
#                     documents.extend(text_docs)
#         except Exception as e:
#             logger.error(f"Text extraction failed: {str(e)}")
        
#         return documents

# class SCADGenerator:
#     """Generate OpenSCAD code using RAG and LLMs"""
    
#     def __init__(self):
#         self.prompt_template = ChatPromptTemplate.from_messages([
#             ("system", (
#                 "You are a CAD engineering assistant with a deep understanding of design requirements. "
#                 "You have the ability to interpret very generic prompts and make all necessary assumptions when details are missing. "
#                 "Generate syntax and semantic valid OpenSCAD code based on user requirements with the following guidelines:\n\n"
#                 "1. Use metric units (mm)\n"
#                 "2. Follow OpenSCAD best practices\n"
#                 "3. Include parameters for customization\n"
#                 "4. Add clear comments explaining key sections and design decisions\n"
#                 "5. Use {name} as the module name\n"
#                 "6. Thoroughly analyze the prompt and assume any missing details logically\n"
#                 "7. Return only the SCAD code without markdown formatting. "
#                 "Perform introspective research over the uploaded documents for syntax, types of parameters, and all such details then provide complete, robust, and fully working code."
#             )),
#             ("user", "{query}")
#         ])
        
#         self.llm = self._initialize_llm()
#         self.embeddings = self._initialize_embeddings()
#         self.refinement_llm = self._initialize_refinement_llm()
        
#     def _initialize_llm(self):
#         """Initialize Groq LLM with fallback"""
#         try:
#             return ChatGroq(
#                 temperature=0.3,
#                 model_name="llama-3.3-70b-versatile",
#                 api_key=os.getenv("GROQ_API_KEY")
#             )
#         except Exception as e:
#             logger.error(f"LLM initialization failed: {str(e)}")
#             return None
            
#     def _initialize_refinement_llm(self):
#         """Initialize the second LLM for code refinement"""
#         try:
#             return ChatGroq(
#                 temperature=0.2,
#                 model_name="deepseek-r1-distill-llama-70b",
#                 api_key=os.getenv("GROQ_API_KEY")
#             )
#         except Exception as e:
#             logger.error(f"Refinement LLM initialization failed: {str(e)}")
#             return None
            
#     def _initialize_embeddings(self):
#         """Initialize embeddings with fallback"""
#         try:
#             return GoogleGenerativeAIEmbeddings(
#                 model="models/embedding-001",
#                 api_key=os.getenv("GOOGLE_API_KEY")
#             )
#         except Exception as e:
#             logger.error(f"Embeddings initialization failed: {str(e)}")
#             return None
    
#     def retrieve_designs(self, query: str, k: int = 3) -> List[Document]:
#         """Retrieve similar designs from vector store"""
#         if not self.embeddings:
#             return []
            
#         try:
#             return Chroma(
#                 persist_directory=VECTOR_STORE_DIR,
#                 embedding_function=self.embeddings,
#                 collection_name="cad_designs",
#                 client_settings=ChromaSettings(
#                     chroma_db_impl="duckdb+parquet",
#                     anonymized_telemetry=False
#                 )
#             ).similarity_search(query, k=k)
#         except Exception as e:
#             logger.error(f"Vector store retrieval failed: {str(e)}")
#             return []
    
#     def refine_code(self, query: str, initial_code: str, context: str, name: str) -> str:
#         """Refine the generated SCAD code using a second LLM"""
#         try:
#             refinement_prompt = ChatPromptTemplate.from_messages([
#                 ("system", (
#                     "You are an expert OpenSCAD code reviewer. Your job is to analyze OpenSCAD code and improve it for robustness, syntax correctness, and complete alignment with the user's design requirements. "
#                     "Consider also the context provided from related documents analysis and vector store data. "
#                     "Return only the final refined OpenSCAD code without any explanations or markdown formatting."
#                 )),
#                 ("user", (
#                     "User design query: {query}\n\n"
#                     "Context from documents: {context}\n\n"
#                     "Initial generated code:\n\n{initial_code}\n\n"
#                     "Please refine the above code to perfection."
#                 ))
#             ])
            
#             chain = refinement_prompt | self.refinement_llm | StrOutputParser()
#             refined_code = chain.invoke({
#                 "query": query,
#                 "context": context,
#                 "initial_code": initial_code,
#                 "name": name
#             })
#             return refined_code
#         except Exception as e:
#             logger.error(f"Code refinement failed: {str(e)}")
#             return initial_code

#     def generate_code(self, query: str, name: str = "GeneratedDesign") -> str:
#         """Generate SCAD code using RAG pipeline with an extra refinement pass"""
#         try:
#             context_docs = self.retrieve_designs(query)
#             context = "\n\n".join([doc.page_content for doc in context_docs])
            
#             if self.llm:
#                 chain = self.prompt_template | self.llm | StrOutputParser()
#                 generated_code = chain.invoke({
#                     "context": context,
#                     "query": query,
#                     "name": name
#                 })
#             else:
#                 generated_code = f"module {name}() {{\n    // Error: Primary LLM not available\n}}"
            
#             # Call second LLM for refinement
#             if self.refinement_llm:
#                 refined_code = self.refine_code(query, generated_code, context, name)
#                 return refined_code
#             else:
#                 return generated_code
#         except Exception as e:
#             logger.error(f"Code generation failed: {str(e)}")
#             return f"module {name}() {{\n    // Error: {str(e)}\n}}"

# def initialize_session_state():
#     """Initialize Streamlit session state variables"""
#     defaults = {
#         "vector_store": None,
#         "processed_files": set(),
#         "scad_code": None
#     }
#     for key, value in defaults.items():
#         if key not in st.session_state:
#             st.session_state[key] = value

# def save_to_vector_store(docs: List[Document]):
#     """Create or update Chroma vector store"""
#     try:
#         embeddings = GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001",
#             api_key=os.getenv("GOOGLE_API_KEY")
#         )
        
#         if os.path.exists(VECTOR_STORE_DIR):
#             vector_store = Chroma(
#                 persist_directory=VECTOR_STORE_DIR,
#                 embedding_function=embeddings,
#                 collection_name="cad_designs",
#                 client_settings=ChromaSettings(
#                     embedding_function_redownload=True,
#                     anonymized_telemetry=False
#                 )
#             )
#             vector_store.add_documents(docs)
#         else:
#             vector_store = Chroma.from_documents(
#                 documents=docs,
#                 embedding=embeddings,
#                 persist_directory=VECTOR_STORE_DIR,
#                 collection_name="cad_designs",
#                 client_settings=ChromaSettings(
#                     chroma_db_impl="duckdb+parquet",
#                     anonymized_telemetry=False
#                 )
#             )
            
#         vector_store.persist()
#         return vector_store
#     except Exception as e:
#         logger.error(f"Vector store save failed: {str(e)}")
#         return None

# def main():
#     """Main Streamlit application"""
#     st.set_page_config(
#         page_title="CAD Design Generator",
#         page_icon="🖨️",
#         layout="wide"
#     )
    
#     initialize_session_state()
#     processor = CADVisionProcessor()
#     scad_gen = SCADGenerator()
    
#     st.title("AI-Powered CAD Design Generator")
    
#     with st.sidebar:
#         st.header("Training Data Management")
#         uploaded_files = st.file_uploader(
#             "Upload CAD PDFs",
#             type=SUPPORTED_EXTS,
#             accept_multiple_files=True
#         )
        
#         if uploaded_files and st.button("Process Documents"):
#             with st.spinner("Analyzing CAD documents..."):
#                 try:
#                     new_docs = []
#                     for file in uploaded_files:
#                         if file.name in st.session_state.processed_files:
#                             continue
                            
#                         with tempfile.NamedTemporaryFile(delete=False) as tmp:
#                             tmp.write(file.getvalue())
#                             docs = processor.process_pdf(tmp.name)
#                             new_docs.extend(docs)
                            
#                         st.session_state.processed_files.add(file.name)
                    
#                     if new_docs:
#                         save_to_vector_store(new_docs)
#                         st.success(f"Processed {len(new_docs)} document chunks!")
#                 except Exception as e:
#                     st.error(f"Processing error: {str(e)}")
        
#         st.divider()
#         st.markdown("### System Status")
#         if os.path.exists(VECTOR_STORE_DIR):
#             st.success("Vector store ready")
#         else:
#             st.info("Upload PDFs to initialize")
    
#     tab1, tab2 = st.tabs(["Design Generator", "Knowledge Base"])
    
#     with tab1:
#         st.header("Generate New Design")
        
#         col1, col2 = st.columns([3, 2])
#         with col1:
#             design_query = st.text_area(
#                 "Design Requirements:",
#                 height=200,
#                 placeholder="Create a hemisphere with outer radius 10mm and inner radius 5mm"
#             )
#             design_name = st.text_input("Module Name:", value="CustomDesign")
            
#             if st.button("Generate SCAD Code"):
#                 if design_query.strip():
#                     with st.spinner("Engineering your design..."):
#                         st.session_state.scad_code = scad_gen.generate_code(design_query, design_name)
#                         st.rerun()
#                 else:
#                     st.warning("Please enter design requirements")
        
#         with col2:
#             if st.session_state.scad_code:
#                 st.download_button(
#                     label="Download SCAD File",
#                     data=st.session_state.scad_code,
#                     file_name=f"{design_name}.scad",
#                     mime="text/x-openscad"
#                 )
#                 st.divider()
#                 st.subheader("Generated Code")
#                 st.code(st.session_state.scad_code, language="openscad")
    
#     with tab2:
#         st.header("Knowledge Base Explorer")
#         if os.path.exists(VECTOR_STORE_DIR):
#             search_query = st.text_input("Search knowledge base:")
#             k_results = st.slider("Results to show", 1, 10, 3)
            
#             if search_query:
#                 try:
#                     results = Chroma(
#                         persist_directory=VECTOR_STORE_DIR,
#                         embedding_function=GoogleGenerativeAIEmbeddings(),
#                         collection_name="cad_designs",
#                         client_settings=ChromaSettings(
#                             chroma_db_impl="duckdb+parquet",
#                             anonymized_telemetry=False
#                         )
#                     ).similarity_search(search_query, k=k_results)
                    
#                     for idx, doc in enumerate(results):
#                         with st.expander(f"Result {idx+1} from {Path(doc.metadata['source']).name}"):
#                             st.json(doc.metadata)
#                             st.text(doc.page_content[:500] + "...")
#                 except Exception as e:
#                     st.error(f"Search failed: {str(e)}")
#         else:
#             st.info("Process PDFs to populate knowledge base")

# if __name__ == "__main__":
#     main()

import os
import base64
import tempfile
import logging
import shutil  # For checking if poppler is installed
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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Use FAISS as the vector store backend instead of Chroma
from langchain.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_EXTS = [".pdf"]
FAISS_INDEX_DIR = "faiss_index"
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
        """Extract page images from PDF using pdf2image and compress if they are too large"""
        images = []
        MAX_IMAGE_SIZE = 500 * 1024  # 500 KB threshold
        MAX_PIXEL_COUNT = 33177600  # Maximum allowed pixels

        # Check if poppler is installed (pdftoppm must be in PATH)
        if not shutil.which("pdftoppm"):
            logger.error("Poppler is not installed or not in PATH. Please install poppler to enable image extraction.")
            return images  # Return empty list if poppler is not available

        try:
            pil_images = convert_from_path(pdf_path)
            for page_num, pil_image in enumerate(pil_images):
                # Check and resize if the image exceeds the allowed pixel count
                w, h = pil_image.size
                if w * h > MAX_PIXEL_COUNT:
                    factor = (MAX_PIXEL_COUNT / (w * h)) ** 0.5
                    new_size = (int(w * factor), int(h * factor))
                    logger.info(f"Resizing page {page_num + 1} image from {pil_image.size} to {new_size} to meet pixel limit.")
                    # Use the updated resampling method
                    pil_image = pil_image.resize(new_size, resample=Image.Resampling.LANCZOS)

                try:
                    # Save the image to a BytesIO buffer in JPEG format
                    img_byte_arr = BytesIO()
                    pil_image.save(img_byte_arr, format="JPEG")
                    image_bytes = img_byte_arr.getvalue()

                    # If the image is larger than the threshold, recompress it with lower quality
                    if len(image_bytes) > MAX_IMAGE_SIZE:
                        logger.info(f"Page {page_num + 1} image ({len(image_bytes)} bytes) exceeds threshold. Recompressing...")
                        compressed_arr = BytesIO()
                        pil_image.save(compressed_arr, format="JPEG", quality=70, optimize=True)
                        image_bytes = compressed_arr.getvalue()
                        logger.info(f"Compressed image size: {len(image_bytes)} bytes")
                except Exception as inner_e:
                    logger.error(f"Error processing image for page {page_num + 1}: {str(inner_e)}")
                    continue

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
                model="llama-3.2-11b-vision-preview",
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
                temperature=0.5,
                max_tokens=5000
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
                "You are a CAD engineering assistant with a deep understanding of design requirements. "
                "You have the ability to interpret very generic prompts and make all necessary assumptions when details are missing. "
                "Generate syntax and semantic valid OpenSCAD code based on user requirements with the following guidelines:\n\n"
                "1. Use metric units (mm)\n"
                "2. Follow OpenSCAD best practices\n"
                "3. Include parameters for customization\n"
                "4. Add clear comments explaining key sections and design decisions\n"
                "5. Use {name} as the module name\n"
                "6. Thoroughly analyze the prompt and assume any missing details logically\n"
                "7. Return only the SCAD code without markdown formatting. "
                "Perform introspective research over the uploaded documents for syntax, types of parameters, and all such details then provide complete, robust, and fully working code."
            )),
            ("user", "{query}")
        ])
        
        self.llm = self._initialize_llm()
        self.embeddings = self._initialize_embeddings()
        self.refinement_llm = self._initialize_refinement_llm()
        
    def _initialize_llm(self):
        """Initialize Groq LLM with fallback"""
        try:
            return ChatGroq(
                temperature=0.3,
                model_name="llama-3.3-70b-versatile",
                api_key=os.getenv("GROQ_API_KEY")
            )
        except Exception as e:
            logger.error(f"LLM initialization failed: {str(e)}")
            return None
            
    def _initialize_refinement_llm(self):
        """Initialize the second LLM for code refinement"""
        try:
            return ChatGroq(
                temperature=0.2,
                model_name="deepseek-r1-distill-llama-70b",
                api_key=os.getenv("GROQ_API_KEY")
            )
        except Exception as e:
            logger.error(f"Refinement LLM initialization failed: {str(e)}")
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
        """Retrieve similar designs from FAISS vector store"""
        if not self.embeddings:
            return []
            
        try:
            if os.path.exists(FAISS_INDEX_DIR):
                vector_store = FAISS.load_local(FAISS_INDEX_DIR, self.embeddings)
                return vector_store.similarity_search(query, k=k)
            else:
                return []
        except Exception as e:
            logger.error(f"Vector store retrieval failed: {str(e)}")
            return []
    
    def refine_code(self, query: str, initial_code: str, context: str, name: str) -> str:
        """Refine the generated SCAD code using a second LLM"""
        try:
            refinement_prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are an expert OpenSCAD code reviewer. Your job is to analyze OpenSCAD code and improve it for robustness, syntax correctness, and complete alignment with the user's design requirements. "
                    "Consider also the context provided from related documents analysis and vector store data. "
                    "Return only the final refined OpenSCAD code without any explanations or markdown formatting."
                )),
                ("user", (
                    "User design query: {query}\n\n"
                    "Context from documents: {context}\n\n"
                    "Initial generated code:\n\n{initial_code}\n\n"
                    "Please refine the above code to perfection."
                ))
            ])
            
            chain = refinement_prompt | self.refinement_llm | StrOutputParser()
            refined_code = chain.invoke({
                "query": query,
                "context": context,
                "initial_code": initial_code,
                "name": name
            })
            return refined_code
        except Exception as e:
            logger.error(f"Code refinement failed: {str(e)}")
            return initial_code

    def generate_code(self, query: str, name: str = "GeneratedDesign") -> str:
        """Generate SCAD code using RAG pipeline with an extra refinement pass"""
        try:
            context_docs = self.retrieve_designs(query)
            context = "\n\n".join([doc.page_content for doc in context_docs])
            
            if self.llm:
                chain = self.prompt_template | self.llm | StrOutputParser()
                generated_code = chain.invoke({
                    "context": context,
                    "query": query,
                    "name": name
                })
            else:
                generated_code = f"module {name}() {{\n    // Error: Primary LLM not available\n}}"
            
            # Call second LLM for refinement
            if self.refinement_llm:
                refined_code = self.refine_code(query, generated_code, context, name)
                return refined_code
            else:
                return generated_code
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
    """Create or update FAISS vector store"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        if os.path.exists(FAISS_INDEX_DIR):
            vector_store = FAISS.load_local(FAISS_INDEX_DIR, embeddings)
            vector_store.add_documents(docs)
        else:
            vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(FAISS_INDEX_DIR)
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
        if os.path.exists(FAISS_INDEX_DIR):
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
        if os.path.exists(FAISS_INDEX_DIR):
            search_query = st.text_input("Search knowledge base:")
            k_results = st.slider("Results to show", 1, 10, 3)
            
            if search_query:
                try:
                    embedding = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        api_key=os.getenv("GOOGLE_API_KEY")
                    )
                    vector_store = FAISS.load_local(FAISS_INDEX_DIR, embedding)
                    results = vector_store.similarity_search(search_query, k=k_results)
                    
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
