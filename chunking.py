"""Split parsed document elements into heading- and table-aware passages."""
import re
import textwrap


STRONG = re.compile(
    r"^(?:\d+(?:\.\d+){0,2}[.)]?\s+[A-Za-z(]"
    r"|[A-Z][A-Z0-9 ,\-/&:()\u2013.]{7,}:?$"
    r"|\([ivx]+\)\s+[A-Z]"
    r"|[a-z]\.?\)\s+[A-Z])"
)


def _table_rows(element):
    """Return each table row as one indivisible text unit."""
    for row in element.get("rows", []):
        cells = []
        for cell in row:
            if isinstance(cell, dict):
                text = cell.get("text", "")
            else:
                text = cell or ""
            cells.extend(line.strip() for line in text.splitlines() if line.strip())
        if cells:
            yield "---" if cells == ["---"] else "\n".join([*cells, "---"])


def _units(elements):
    """Flatten ordered layout elements while preserving paragraph and row boundaries."""
    for element in elements:
        if element.get("type") == "table":
            yield from ((re.sub(r"\s+", " ", row), "table-row") for row in _table_rows(element))
        else:
            for line in element.get("text", "").splitlines():
                if line.strip():
                    yield re.sub(r"\s+", " ", line).strip(), "paragraph"


def _split_long_row(row, max_chars):
    """Split an oversized row only at cell/line boundaries where possible."""
    parts, current, size = [], [], 0
    for line in row.splitlines():
        line_parts = [line]
        if len(line) > max_chars:
            line_parts = textwrap.wrap(
                line, width=max_chars, break_long_words=False, break_on_hyphens=False,
            ) or [line]
        for part in line_parts:
            if current and size + len(part) + 1 > max_chars:
                parts.append("\n".join(current))
                current, size = [], 0
            current.append(part)
            size += len(part) + 1
    if current:
        parts.append("\n".join(current))
    return parts


def chunk_elements(elements, min_chars=500, max_chars=1400, row_flush=700, overlap=2):
    """Chunk parsed layout, keeping table rows together and splitting before headings."""
    chunks, current, size = [], [], 0

    def flush(carry=0):
        nonlocal current, size
        if current:
            chunks.append("\n".join(text for text, _ in current))
            current = current[-carry:] if carry else []
            size = sum(len(text) + 1 for text, _ in current)

    for text, kind in _units(elements):
        if kind == "table-row":
            if size >= row_flush:
                flush()
            if len(text) > max_chars:
                flush()
                parts = _split_long_row(text, max_chars)
                for part in parts[:-1]:
                    current.append((part, kind))
                    size += len(part) + 1
                    flush()
                if parts:
                    current.append((parts[-1], kind))
                    size += len(parts[-1]) + 1
            else:
                if size + len(text) > max_chars and current:
                    flush()
                current.append((text, kind))
                size += len(text) + 1
            continue

        if STRONG.match(text) and size >= min_chars:
            flush()
        if size + len(text) > max_chars and current:
            flush(carry=overlap)
        current.append((text, kind))
        size += len(text) + 1

    flush()
    return chunks
