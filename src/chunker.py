import re
import json
import random
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_f1_rule_chunks() -> list[Document]:
    """
    Reads assembled markdown files and harvests rules into LangChain Documents
    with rich metadata extracted from the assembly tags.
    """
    processed_dir = Path("data/processed")
    chunked_dir = Path("data/chunked")

    if not processed_dir.exists():
        print(
            f"ERROR: Processed directory '{processed_dir}' not found. Run doc_assembler.py first."
        )
        return []

    chunked_dir.mkdir(parents=True, exist_ok=True)

    observed_chunks_path = chunked_dir / "chunks.md"
    jsonl_output_path = chunked_dir / "chunks.jsonl"

    documents = []

    for md_file in processed_dir.glob("*_processed.md"):
        print(f"CHUNKER: Processing assembled file {md_file.name}")

        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        raw_chunks = re.split(r"\n(?=### Section )", content)

        for chunk in raw_chunks:
            chunk = chunk.strip()
            if not chunk or "\n" not in chunk:
                continue

            file_match = re.search(r"SOURCE_FILE:\s+(.*)", chunk)
            page_match = re.search(r"SOURCE_PAGE:\s+(\d+)", chunk)

            source_filename = (
                file_match.group(1).strip() if file_match else md_file.name
            )
            page_val = int(page_match.group(1)) if page_match else None

            lines = chunk.split("\n")
            breadcrumb_line = lines[0].replace("### ", "").strip()

            remaining_text = "\n".join(lines[1:])
            clean_rule_text = re.sub(r"SOURCE_FILE:.*\n?", "", remaining_text)
            clean_rule_text = re.sub(r"SOURCE_PAGE:.*\n?", "", clean_rule_text).strip()

            rule_id_match = re.search(r"\*\*([A-Z]\d+\.\d+\.\d+[a-z]?)\*\*", chunk)
            rule_id_val = rule_id_match.group(1) if rule_id_match else "Unknown"

            parts = [p.strip() for p in breadcrumb_line.split(">")]

            metadata = {
                "source_pdf": source_filename,
                "regulatory_year": 2026,
                "page_number": page_val,
                "rule_id": rule_id_val,
                "breadcrumb": breadcrumb_line,
            }

            if len(parts) >= 3:
                metadata.update(
                    {"section": parts[0], "article": parts[1], "sub_article": parts[2]}
                )

            llm_ready_content = f"[{breadcrumb_line}]\n{clean_rule_text}"
            documents.append(
                Document(page_content=llm_ready_content, metadata=metadata)
            )

    print(f"INFO: Successfully generated {len(documents)} atomic rule chunks.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000, chunk_overlap=300, separators=["\n\n", "\n", "|", " ", ""]
    )

    final_docs = text_splitter.split_documents(documents)
    print(f"INFO: Post-splitting total chunks: {len(final_docs)}")

    print(f"INFO: Saving chunk observations to {observed_chunks_path}")
    with open(observed_chunks_path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(final_docs):
            f.write(f"## CHUNK {i+1}\n")
            f.write(
                f"**Metadata:**\n```json\n{json.dumps(doc.metadata, indent=2)}\n```\n\n"
            )
            f.write(f"**Content:**\n{doc.page_content}\n\n")
            f.write("---\n\n")

    print(f"INFO: Exporting chunks to JSONL: {jsonl_output_path}")
    with open(jsonl_output_path, "w", encoding="utf-8") as f:
        for doc in final_docs:
            # Convert Document object to a simple dictionary
            record = {"page_content": doc.page_content, "metadata": doc.metadata}
            f.write(json.dumps(record) + "\n")

    return final_docs


if __name__ == "__main__":
    docs = create_f1_rule_chunks()

    if docs:
        print("\n" + "=" * 60)
        print("SYSTEM LOG: RANDOM SAMPLE OF 10 CHUNKS")
        print("=" * 60)

        sample_size = min(10, len(docs))
        random_docs = random.sample(docs, sample_size)

        for i, doc in enumerate(random_docs):
            print(f"\n[SAMPLE {i+1}/{sample_size}]")
            print(f"METADATA:\n{json.dumps(doc.metadata, indent=2)}")
            print("\nCONTENT:")
            preview = (
                doc.page_content[:400] + " ... [TRUNCATED]"
                if len(doc.page_content) > 400
                else doc.page_content
            )
            print(preview)
            print("-" * 60)
