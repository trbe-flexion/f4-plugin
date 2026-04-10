"""Text extraction from PDF and DOCX files."""

from __future__ import annotations

from pathlib import Path


def extract_text(file_path: str) -> str:
    """Extract text from a PDF or DOCX file.

    Raises ValueError for unsupported file types.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(file_path)
    elif suffix == ".docx":
        return _extract_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _extract_pdf(file_path: str) -> str:
    import pdfplumber

    pages = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def _extract_docx(file_path: str) -> str:
    import docx

    doc = docx.Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def extract_and_combine(file_paths: list[str]) -> str:
    """Extract text from multiple files and combine with labeled headers.

    Matches opp-capture's multi-document concatenation format.
    """
    if not file_paths:
        return ""

    if len(file_paths) == 1:
        return extract_text(file_paths[0])

    doc_count = len(file_paths)
    header = f"MULTI-DOCUMENT PACKAGE ({doc_count} documents)\n\n"
    sections = []

    for i, file_path in enumerate(file_paths, start=1):
        filename = Path(file_path).name
        separator = "=" * 40
        text = extract_text(file_path)
        section = f"{separator}\nDOCUMENT {i} of {doc_count}: {filename}\n{separator}\n\n{text}"
        sections.append(section)

    return header + "\n\n".join(sections)
