"""
ingest.py — Knowledge Base Ingestion for Self-RAG University Advisory Agent
Loads all 5 university PDF documents, applies structure-aware chunking,
attaches meaningful metadata, and persists to a FAISS vector index.
Uses Mistral embeddings (mistral-embed model).
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "Final" / "Data_share"
FAISS_DIR = str(Path(__file__).parent / "university_kb")

PDF_METADATA_MAP: dict[str, dict] = {
    "CS_Department_Catalog.pdf": {
        "department": "Computer Science",
        "doc_type": "course_catalog",
        "course_level": "undergraduate_graduate",
    },
    "EE_Department_Catalog.pdf": {
        "department": "Electrical Engineering",
        "doc_type": "course_catalog",
        "course_level": "undergraduate",
    },
    "BBA_Department_Catalog (1).pdf": {
        "department": "Business Administration",
        "doc_type": "course_catalog",
        "course_level": "undergraduate",
    },
    "University_Academic_Policies.pdf": {
        "department": "University",
        "doc_type": "academic_policies",
        "course_level": "all",
    },
    "Faculty_Directory.pdf": {
        "department": "University",
        "doc_type": "faculty_directory",
        "course_level": "all",
    },
}


def _get_embeddings() -> MistralAIEmbeddings:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError("MISTRAL_API_KEY not found in .env")
    return MistralAIEmbeddings(model="mistral-embed", api_key=api_key)


def build_knowledge_base(force_rebuild: bool = False) -> FAISS:
    faiss_path = Path(FAISS_DIR)

    if faiss_path.exists() and not force_rebuild:
        print(f"[INGEST] Loading existing FAISS index from {FAISS_DIR}")
        return FAISS.load_local(
            FAISS_DIR,
            _get_embeddings(),
            allow_dangerous_deserialization=True,
        )

    print("[INGEST] Building knowledge base from scratch...")
    all_chunks = []

    for filename, metadata in PDF_METADATA_MAP.items():
        pdf_path = DATA_DIR / filename
        if not pdf_path.exists():
            print(f"  [WARN] Not found, skipping: {pdf_path}")
            continue

        print(f"  Loading: {filename}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for page in pages:
            page.metadata.update(metadata)
            page.metadata["source_file"] = filename

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n\n", "\n\n", "\n", ". ", ", ", " ", ""],
            length_function=len,
        )

        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)
        print(f"    -> {len(pages)} pages -> {len(chunks)} chunks")

    if not all_chunks:
        print("[ERROR] No documents loaded. Check DATA_DIR path.")
        sys.exit(1)

    print(f"\n[INGEST] Total chunks: {len(all_chunks)}")
    print("[INGEST] Embedding with Mistral and indexing into FAISS...")

    vectorstore = FAISS.from_documents(all_chunks, _get_embeddings())
    faiss_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(FAISS_DIR)

    print(f"[INGEST] Done. {len(all_chunks)} chunks indexed and saved to {FAISS_DIR}")
    return vectorstore


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild the index.")
    args = parser.parse_args()

    vs = build_knowledge_base(force_rebuild=args.rebuild)
    print(f"\n[DONE] FAISS index ready at {FAISS_DIR}")
