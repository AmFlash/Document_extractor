"""Read PDF, DOCX, and DOC tender files into raw and cleaned text."""
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import pdfplumber
from docx import Document
from docx.document import Document as _Document
from docx.oxml.ns import qn
from docx.table import Table


def _pdf_page_elements(page, page_number):
    """Extract ordered text and table elements from one PDF page."""
    try:
        tables = page.find_tables()
    except Exception:
        tables = []
    boxes = [t.bbox for t in tables]

    def outside(obj):
        if obj.get("object_type") != "char":
            return True
        x = (obj["x0"] + obj["x1"]) / 2
        y = (obj["top"] + obj["bottom"]) / 2
        return not any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in boxes)

    body = page.filter(outside) if boxes else page
    items = []
    for ln in body.extract_text_lines():
        text = ln["text"].strip()
        if text:
            items.append((ln["top"], {
                "type": "paragraph", "page": page_number, "text": text,
            }))
    for table in tables:
        rows = []
        text_lines = []
        for raw_row in table.extract():
            cells = [cell.strip() for cell in (raw_row or []) if cell and cell.strip()]
            if not cells:
                continue
            for cell in cells:
                text_lines += [line.strip() for line in cell.splitlines() if line.strip()]
            text_lines.append("---")
            rows.append(cells)
        if text_lines:
            items.append((table.bbox[1], {
                "type": "table", "page": page_number,
                "bbox": list(table.bbox), "rows": rows, "text_lines": text_lines,
            }))
    items.sort(key=lambda item: item[0])
    return [element for _, element in items]


def parse_pdf_document(path):
    raw_pages, page_elements = [], []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            raw_pages.append(text)
            try:
                page_elements.append(_pdf_page_elements(page, page_number))
            except Exception:
                page_elements.append([
                    {"type": "paragraph", "page": page_number, "text": line.strip()}
                    for line in text.splitlines() if line.strip()
                ])

    raw = "\n".join(raw_pages)
    if len(raw.strip()) < 80 * max(1, len(page_elements)):
        raise ValueError("this PDF has almost no text (scanned?). Run OCR first, e.g. "
                         "'ocrmypdf in.pdf out.pdf', then run this script on out.pdf")

    key = lambda line: re.sub(r"\d+", "#", line)
    counts = Counter(
        key(line)
        for elements in page_elements
        for line in {line for element in elements for line in _element_lines(element)}
    )
    page_count = len(page_elements)
    threshold = max(3, page_count // 2)
    clean_lines, elements = [], []
    for page in page_elements:
        for element in page:
            lines = []
            for line in _element_lines(element):
                if (line != "---" and page_count >= 4 and len(line) >= 4
                        and counts[key(line)] >= threshold):
                    continue
                if line == "---" and clean_lines and clean_lines[-1] == "---":
                    continue
                clean_lines.append(line)
                lines.append(line)
            if lines:
                saved = dict(element)
                if element["type"] == "table":
                    saved["text_lines"] = lines
                else:
                    saved["text"] = lines[0]
                elements.append(saved)
    return raw, "\n".join(clean_lines), elements


def _children(parent):
    if isinstance(parent, _Document):
        return parent.element.body.iterchildren()
    return parent._tc.iterchildren()


def _iter_blocks(elements):
    """Paragraphs and tables, including those inside Word content controls."""
    for child in elements:
        if child.tag == qn("w:sdt"):
            content = child.find(qn("w:sdtContent"))
            if content is not None:
                yield from _iter_blocks(content.iterchildren())
        else:
            yield child


def _para_text(paragraph):
    """Paragraph text including tracked insertions; deleted text is excluded."""
    out = []
    for element in paragraph.iter():
        if element.tag == qn("w:t"):
            out.append(element.text or "")
        elif element.tag == qn("w:tab") and element.getparent().tag == qn("w:r"):
            out.append(" ")
    return "".join(out).strip()


def _render_elements(parent):
    elements = []
    for child in _iter_blocks(_children(parent)):
        if child.tag == qn("w:p"):
            text = _para_text(child)
            if text:
                elements.append({"type": "paragraph", "text": text})
        elif child.tag == qn("w:tbl"):
            elements.append(_render_table_element(Table(child, parent)))
    return elements


def _element_lines(element):
    if element["type"] == "table":
        return element["text_lines"]
    return [element["text"]]


def _render_table_element(table):
    rows, text_lines, seen = [], [], set()
    for row in table.rows:
        row_lines, row_cells = [], []
        for cell in row.cells:
            if cell._tc in seen:
                continue
            seen.add(cell._tc)
            cell_elements = _render_elements(cell)
            cell_lines = [line for element in cell_elements for line in _element_lines(element)]
            row_lines.extend(cell_lines)
            row_cells.append({"text": "\n".join(cell_lines), "elements": cell_elements})
        if row_lines:
            text_lines.extend(row_lines)
            text_lines.append("---")
            rows.append(row_cells)
    return {"type": "table", "rows": rows, "text_lines": text_lines}


def parse_docx_document(path):
    doc = Document(path)
    elements = _render_elements(doc)
    body = [line for element in elements for line in _element_lines(element)]
    head_foot = []
    for section in doc.sections:
        for part in (section.header, section.first_page_header, section.footer):
            try:
                head_foot += [p.text.strip() for p in part.paragraphs if p.text.strip()]
                for table in part.tables:
                    head_foot += [cell.text.strip() for row in table.rows
                                  for cell in row.cells if cell.text.strip()]
            except Exception:
                pass
    clean = "\n".join(body)
    raw = "\n".join(head_foot + body)
    return raw, clean, elements


def doc_to_docx(path):
    """Convert old .doc files using LibreOffice."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise ValueError(".doc files need LibreOffice installed, or save the file as .docx first")
    out = tempfile.mkdtemp()
    subprocess.run([soffice, "--headless", "--convert-to", "docx", "--outdir", out, str(path)],
                   check=True, timeout=180, capture_output=True)
    return str(Path(out) / (Path(path).stem + ".docx"))


def load_document(path):
    """Return raw text, cleaned text, and ordered parsed elements."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return parse_pdf_document(path)
    if ext == ".docx":
        return parse_docx_document(path)
    if ext == ".doc":
        return parse_docx_document(doc_to_docx(path))
    raise ValueError(f"unsupported file type: {ext}")
