import json
from pathlib import Path

import httpx
from dotenv import load_dotenv
from llama_cloud import LlamaCloud

load_dotenv()

raw_dir = Path("data/raw")
parsed_base_dir = Path("data/parsed")

raw_dir.mkdir(parents=True, exist_ok=True)
parsed_base_dir.mkdir(parents=True, exist_ok=True)

client = LlamaCloud()

fia_custom_prompt = """
This is a highly structured FIA technical regulation. 
1. STRIKETHROUGH: Do NOT capture any text that has a strikethrough line through it. These represent deleted rules.
2. HIERARCHY: Maintain the legal numbering (e.g., A2.2.2).
3. TABLES: Ensure the 'Points' tables are perfectly reconstructed. If a cell is empty in the PDF, leave it empty in the output.
"""

pdf_files = list(raw_dir.glob("*.pdf"))

if not pdf_files:
    print(
        f"No PDF files found in {raw_dir.resolve()}. Drop some files in there and run again."
    )
else:
    print(f"Found {len(pdf_files)} PDF(s) to process. Starting batch extraction...\n")

    for pdf_path in pdf_files:
        filename = pdf_path.stem
        print(f"Processing: {pdf_path.name}")

        doc_dir = parsed_base_dir / filename
        doc_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = client.parsing.parse(
                upload_file=str(pdf_path),
                tier="agentic",
                version="latest",
                crop_box={"top": 0.07, "bottom": 0.08},
                agentic_options={"custom_prompt": fia_custom_prompt},
                output_options={
                    "images_to_save": ["layout"],
                    "markdown": {
                        "tables": {
                            "compact_markdown_tables": True,
                            "output_tables_as_markdown": True,
                        },
                    },
                },
                expand=["markdown", "items", "images_content_metadata"],
            )

            json_file_path = doc_dir / f"{filename}_parsed.json"
            result_dict = (
                result.model_dump()
                if hasattr(result, "model_dump")
                else getattr(result, "__dict__", {})
            )

            with json_file_path.open("w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)
            print(f"  - Saved JSON: {json_file_path.name}")

            if hasattr(result, "markdown") and result.markdown:
                md_file_path = doc_dir / f"{filename}_parsed.md"
                md_content = "\n\n".join(
                    page.markdown for page in result.markdown.pages
                )
                md_file_path.write_text(md_content + "\n\n", encoding="utf-8")
                print(f"  - Saved Markdown: {md_file_path.name}")

            if metadata_obj := getattr(result, "images_content_metadata", None):

                image_list = (
                    metadata_obj
                    if isinstance(metadata_obj, list)
                    else getattr(
                        metadata_obj,
                        "images",
                        (
                            metadata_obj.get("images", [])
                            if isinstance(metadata_obj, dict)
                            else []
                        ),
                    )
                )

                if image_list:
                    doc_img_dir = doc_dir / "images"
                    doc_img_dir.mkdir(parents=True, exist_ok=True)

                    print(f"  - Downloading {len(image_list)} image(s)...")

                    with httpx.Client() as http_client:
                        for i, image in enumerate(image_list):
                            is_dict = isinstance(image, dict)
                            img_name = (
                                image.get("filename")
                                if is_dict
                                else getattr(image, "filename", f"img_{i}")
                            )
                            url = (
                                image.get("presigned_url")
                                if is_dict
                                else getattr(image, "presigned_url", None)
                            )

                            if url and (res := http_client.get(url)).status_code == 200:
                                (doc_img_dir / img_name).write_bytes(res.content)
                    print(f"  - Images saved to: {doc_img_dir.name}/")

        except Exception as e:
            print(f"  - Error processing: {e}")

        print("-" * 50)

    print("\nBatch processing complete.")
