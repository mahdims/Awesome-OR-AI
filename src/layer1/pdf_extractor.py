import requests
import hashlib
from pathlib import Path
import fitz  # PyMuPDF

CACHE_DIR = Path("cache/pdfs")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def fetch_arxiv_pdf_text(arxiv_id: str) -> tuple[str, str]:
    """
    Download PDF from ArXiv and extract text.
    Returns: (full_text, pdf_hash)

    Args:
        arxiv_id: ArXiv identifier (e.g., "2501.12345")

    Returns:
        tuple: (extracted_text, sha256_hash)
    """
    # Check cache
    cache_file = CACHE_DIR / f"{arxiv_id.replace('/', '_')}.txt"
    hash_file = CACHE_DIR / f"{arxiv_id.replace('/', '_')}.hash"

    if cache_file.exists() and hash_file.exists():
        print(f"  [CACHE HIT] Using cached PDF text for {arxiv_id}")
        return cache_file.read_text(encoding='utf-8'), hash_file.read_text()

    print(f"  [DOWNLOAD] Fetching PDF from ArXiv: {arxiv_id}")

    # Download PDF
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    response = requests.get(pdf_url, timeout=60)
    response.raise_for_status()

    pdf_bytes = response.content
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

    print(f"  [EXTRACT] Extracting text from PDF ({len(pdf_bytes)} bytes)")

    # Extract text using PyMuPDF
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []

        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")

        full_text = "\n\n".join(text_parts)

        print(f"  [SUCCESS] Extracted {len(full_text)} characters from {len(doc)} pages")

        # Cache
        cache_file.write_text(full_text, encoding='utf-8')
        hash_file.write_text(pdf_hash)

        return full_text, pdf_hash

    except Exception as e:
        print(f"  [ERROR] Failed to extract text: {e}")
        raise

def clear_cache():
    """Clear all cached PDF text files."""
    import shutil
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("[CACHE] Cleared PDF cache")
