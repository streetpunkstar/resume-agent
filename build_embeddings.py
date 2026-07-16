import os
import re
import fitz  # PyMuPDF
import pdfplumber
import ollama
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from export_background import export_professional_background

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------
SOURCE_FOLDER = "./source_documents"   # Drop any .pdf, .txt, or .md files in here
DB_FOLDER = "./resume_vectors_db"      # Folder where your vectors will be saved
COLLECTION_NAME = "resume_data"        # Name of the internal database table
SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")
VALID_CATEGORIES = ("professional", "personal")

# Used to detect when a hyperlink's visible anchor text is just the
# candidate's own name (e.g. "LinkedIn: Richard Paasch" where "Richard
# Paasch" is the clickable text) rather than a meaningful label. Previously
# this ambiguity was patched downstream in job_assistant.py's system prompt
# with an explicit warning telling the LLM never to treat a name-like string
# as the candidate's actual name if it appears next to a URL — a fragile fix
# that depends on the model correctly following that instruction every time.
# Fixing it here instead means the link chunk itself is unambiguous, so
# nothing downstream has to compensate for it at all.
CANDIDATE_NAME = "Richard Paasch"

# Matches Markdown-style links: [anchor text](https://...)
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")


def discover_source_files(source_folder):
    """
    Walks the source folder and tags each file with a category based on which
    subfolder it lives in. Expected layout:
        source_documents/professional/...  -> tagged "professional"
        source_documents/personal/...      -> tagged "personal"
    Files dropped directly in source_documents/ (no subfolder) default to
    "professional" to stay safe.
    Returns a list of (file_path, filename, category) tuples.
    """
    discovered = []
    for root, _dirs, files in os.walk(source_folder):
        rel_root = os.path.relpath(root, source_folder)
        if rel_root == ".":
            category = "professional"
        else:
            top_level = rel_root.split(os.sep)[0].lower()
            category = top_level if top_level in VALID_CATEGORIES else "professional"

        for filename in files:
            if filename.lower().endswith(SUPPORTED_EXTENSIONS):
                discovered.append((os.path.join(root, filename), filename, category))

    return discovered


def relabel_name_as_anchor(anchor_text: str, url: str) -> str:
    """
    If the anchor text is just the candidate's own name (case-insensitive,
    ignoring extra whitespace), replace it with a description derived from
    the URL's domain instead — e.g. 'Richard Paasch' linking to
    linkedin.com/in/... becomes 'linkedin.com profile' rather than being
    stored under the ambiguous key 'Richard Paasch'. Leaves genuinely
    descriptive anchor text (e.g. 'my GitHub', 'portfolio') untouched.
    """
    if anchor_text.strip().lower() != CANDIDATE_NAME.strip().lower():
        return anchor_text

    domain_match = re.search(r"https?://(?:www\.)?([^/]+)", url)
    domain = domain_match.group(1) if domain_match else "link"
    return f"{domain} profile"


# ----------------------------------------------------
# TASK 1: FILE PARSING (per file type)
# ----------------------------------------------------
def parse_pdf(file_path):
    """Extracts layout-aware body text and hyperlinks from a PDF."""
    doc = fitz.open(file_path)
    links_found = {}

    # 1. Map out every single link and its exact underlying anchor text
    for page_num in range(len(doc)):
        page = doc[page_num]
        for link in page.links():
            rect = link["from"]
            text_under_link = page.get_text("text", clip=rect).strip()
            if text_under_link and "uri" in link:
                label = relabel_name_as_anchor(text_under_link, link["uri"])
                links_found[label] = link["uri"]

    # 2. Extract layout-aware main body text
    structured_pages = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=True, use_text_flow=True)
            if text:
                structured_pages.append(text)
    full_body_text = "\n\n".join(structured_pages)

    return full_body_text, links_found


def parse_text_file(file_path):
    """Reads a plain .txt or .md file and pulls out any Markdown-style links."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        full_body_text = f.read()

    links_found = {}
    for anchor, url in MARKDOWN_LINK_PATTERN.findall(full_body_text):
        label = relabel_name_as_anchor(anchor.strip(), url.strip())
        links_found[label] = url.strip()

    return full_body_text, links_found


def parse_file(file_path):
    """Routes a file to the right parser based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext in (".txt", ".md"):
        return parse_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ----------------------------------------------------
# TASK 2: TEXT CHUNKING & DATABASE INJECTION
# ----------------------------------------------------
def chunk_document(body_text, links_dict, source_name):
    """Splits one document's body text into chunks, plus a links chunk if any were found."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(body_text)
    chunk_records = [{"text": chunk, "type": "content"} for chunk in chunks]

    if links_dict:
        link_directory_chunk = f"--- VERIFICATION LINKS & EVIDENCE ({source_name}) ---\n"
        for anchor, url in links_dict.items():
            link_directory_chunk += f"Evidence for '{anchor}': {url}\n"
        chunk_records.append({"text": link_directory_chunk, "type": "links"})

    return chunk_records


def build_vector_database(all_documents):
    """
    all_documents: list of (source_name, body_text, links_dict, category) tuples,
    one per successfully parsed file.
    """
    chroma_client = chromadb.PersistentClient(path=DB_FOLDER)

    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    print("[*] Generating local vector embeddings via Ollama...")
    total_chunks = 0

    for source_name, body_text, links_dict, category in all_documents:
        chunk_records = chunk_document(body_text, links_dict, source_name)
        print(f"[*] {source_name} [{category}]: {len(chunk_records)} chunks")

        # Sanitize filename so it's safe to use inside a Chroma document ID
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", source_name)

        for idx, record in enumerate(chunk_records):
            response = ollama.embeddings(model='nomic-embed-text', prompt=record["text"])
            vector = response['embedding']

            collection.add(
                ids=[f"{safe_name}_chunk_{idx}"],
                embeddings=[vector],
                documents=[record["text"]],
                metadatas=[{"source": source_name, "type": record["type"], "category": category}]
            )
            total_chunks += 1

    print(f"[SUCCESS] Database initialized! {total_chunks} vectors from "
          f"{len(all_documents)} file(s) saved to '{DB_FOLDER}'.")


# ----------------------------------------------------
# RUNNER
# ----------------------------------------------------
if __name__ == "__main__":
    if not os.path.isdir(SOURCE_FOLDER):
        print(f"[Error] Create a folder called '{SOURCE_FOLDER}' with 'professional' and "
              f"'personal' subfolders, and drop your files (.pdf, .txt, or .md) into the "
              f"matching one.")
    else:
        source_files = discover_source_files(SOURCE_FOLDER)

        if not source_files:
            print(f"[Error] No supported files found in '{SOURCE_FOLDER}'. "
                  f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}")
        else:
            all_documents = []
            for file_path, filename, category in source_files:
                print(f"[*] Parsing: {filename} [{category}]")
                try:
                    text, links = parse_file(file_path)
                    all_documents.append((filename, text, links, category))
                except Exception as e:
                    print(f"[Warning] Skipping '{filename}' due to error: {e}")

            if all_documents:
                build_vector_database(all_documents)
                print("[*] Re-exporting professional_background.json for job_assistant.py...")
                export_professional_background()
            else:
                print("[Error] No documents were successfully parsed.")
