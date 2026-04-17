import json
import re
from pathlib import Path


def assemble_all_sections():
    """
    Orchestrates the assembly process for all parsed JSON sections.
    """
    parsed_dir = Path("data/parsed")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    for section_folder in parsed_dir.iterdir():
        if not section_folder.is_dir():
            continue

        folder_name = section_folder.name
        json_path = section_folder / f"{folder_name}_parsed.json"

        if not json_path.exists():
            print(f"WARNING: Skipping {folder_name}: Missing source JSON file.")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        original_filename = data.get("job", {}).get("name", "Unknown_Source.pdf")

        output_path = output_dir / f"{folder_name}_processed.md"
        print(f"ASSEMBLER: Building {output_path.name} from {original_filename}...")

        section_letter = folder_name.split("_")[-1].upper()

        assemble_markdown_from_json(
            data, output_path, section_letter, folder_name, original_filename
        )


def assemble_markdown_from_json(
    data, output_path, section_letter, folder_name, original_filename
):
    """
    Performs the actual assembly of text blocks into a structured Markdown document.
    """
    pages = data.get("items", {}).get("pages", [])

    current_section = f"Section {section_letter}"
    current_article = "None"
    current_sub_article = "None"

    in_toc = False
    assembled_lines = []

    article_pattern = rf"(?i)^#*\s*ARTICLE ({section_letter}\d+):?\s*(.*)"
    sub_article_pattern = rf"^#*\s*({section_letter}\d+\.\d+)\s*(.*)"
    # Locate this line in reconstruct_markdown (or assemble_markdown_from_json)
    # Change it to:
    rule_pattern = (
        rf"^(?:#+\s*|\*\*)?({section_letter}\d+\.\d+(?:\.\d+)?[a-z]?)(?:\*\*|:)?"
    )

    for i, page in enumerate(pages, start=1):
        page_num = i
        page_items = page.get("items", [])

        for item in page_items:
            block_text = item.get("md") or item.get("text") or ""
            block_text = block_text.strip()
            item_type = item.get("type", "text")

            if not block_text:
                continue

            # Image Path Remapping (ensuring assets are linked to the assembled file)
            block_text = re.sub(
                r"!\[([^\]]*)\]\(([^)]+)\)",
                lambda m: f"![{m.group(1)}](../parsed/{folder_name}/images/{Path(m.group(2)).name})",
                block_text,
            )

            # Clean up strikethrough text (deleted regulations)
            block_text = re.sub(r"~~.*?~~", "", block_text, flags=re.DOTALL).strip()
            if not block_text:
                continue

            # Skip Table of Contents
            if (
                "CONTENTS:" in block_text.upper()
                or "**CONTENTS**" in block_text.upper()
            ):
                in_toc = True
                continue

            if in_toc:
                if re.search(article_pattern, block_text):
                    in_toc = False
                else:
                    continue

            # Assemble Tables
            if item_type == "table":
                table_md = re.sub(r"\n{2,}", "\n", block_text)
                assembled_lines.append(f"\n\n{table_md}\n")
                continue

            # Clean formatting for hierarchy detection
            clean_text = re.sub(r"[*#]", "", block_text).strip()

            # Identify Articles
            article_match = re.search(article_pattern, clean_text)
            if article_match:
                current_article = f"Article {article_match.group(1)}: {article_match.group(2).strip()}"
                current_sub_article = "None"
                continue

            # Identify Sub-Articles
            sub_article_match = re.match(sub_article_pattern, clean_text)
            if sub_article_match and not re.search(r"\.\d+\.", clean_text):
                current_sub_article = (
                    f"{sub_article_match.group(1)} {sub_article_match.group(2).strip()}"
                )

            # Identify specific Rules and inject assembly metadata
            rule_match = re.search(rule_pattern, clean_text)
            if rule_match:
                rule_id = rule_match.group(1)
                breadcrumb = f"### {current_section} > {current_article} > {current_sub_article} > {rule_id}"

                formatted_rule = (
                    f"\n{breadcrumb}\n"
                    f"SOURCE_FILE: {original_filename}\n"
                    f"SOURCE_PAGE: {page_num}\n"
                    f"\n"
                    f"**{rule_id}** " + block_text.replace(rule_id, "", 1).strip("* :")
                )
                assembled_lines.append(formatted_rule)
                continue

            assembled_lines.append(block_text)

    assembled_content = "\n\n".join(assembled_lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(assembled_content)

    print(f"INFO: Assembly complete for {output_path.name}")


if __name__ == "__main__":
    assemble_all_sections()
