#!/usr/bin/env python3
"""
Tender extractor: PDF / DOCX / DOC -> JSON with exactly these keys

    {
      "name of work": "", "tender no": "", "ato_criteria": "",
      "similar_work_definition": "", "documents_to_be_submitted": [],
      "tech_evaluation_mode": "L1"          # L1, QCBS, or UNKNOWN if the tender never says
    }

- "tender no"            : found with a pattern (no model involved)
- "tech_evaluation_mode" : model answer, double-checked by a pattern on the full text
- other fields           : Gemini model (API key loaded from .env)
- Problems (empty fields, text not found in the source, ...) go to <stem>.warnings.json.

Usage
    pip install -r requirements.txt
    # Add GEMINI_API_KEY=your-key to .env in this folder
    python extract_tender.py "samples/Sample 1.docx" "samples/Sample 2.pdf" "samples/Sample 3.docx"
    python extract_tender.py *.pdf --model gemini-3.6-flash --workers 4
    python extract_tender.py "samples/Sample 2.pdf" --dry-run  # no model call; shows what would be sent
    python extract_tender.py *.pdf --skip-existing         # continue a half-finished batch

Each processed document also saves <stem>.parsed_response.json, <stem>.chunks.json,
and <stem>.tables.json in the output directory.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from chunking import chunk_elements
from document_parser import load_document
from generation import DEFAULT_MODEL, ask_llm
from retriever import retrieve_all, merge_context

HERE = Path(__file__).resolve().parent
PROMPT_FILE = HERE / "extraction_prompt.txt"

AMOUNT_RE = re.compile(r"\d[\d,]{3,}|\d+(?:\.\d+)?\s*(?:crore|cr\b|lac|lakh|million)", re.I)

# ----------------------------------------------------------------------------
# Tender number: pattern only
# ----------------------------------------------------------------------------

TENDER_RE = re.compile(
    r"\b(?:e-?tender|tender|nit|rfp|rfq|rfx|enquiry|bid)\s*"
    r"(?:no\.?|number|ref(?:erence)?\.?(?:\s*no\.?)?)\s*[:\-\u2013.]*\s*"
    r"([A-Za-z0-9][A-Za-z0-9/_\-.()]*[A-Za-z0-9])",
    re.I,
)


def get_tender_no(raw_text):
    """The same number is often written in two ways (cover page vs page header).
    We pick the number that appears most, then write it the way it appears FIRST
    (cover / title block), not the way the repeating page header writes it."""
    hits = []
    for m in TENDER_RE.finditer(raw_text):
        v = m.group(1)
        if re.search(r"\d", v) and re.search(r"[/_\-]", v):
            hits.append((m.start(), v.rstrip(".")))
    if not hits:
        return ""
    canon = lambda s: re.sub(r"[^A-Z0-9]", "", s.upper())
    best_key = Counter(canon(v) for _, v in hits).most_common(1)[0][0]
    return min((pos, v) for pos, v in hits if canon(v) == best_key)[1]

# ----------------------------------------------------------------------------
# L1 or QCBS: a plain text check that backs up the model
# ----------------------------------------------------------------------------

QCBS_RE = re.compile(
    r"\bQCBS\b|quality[\s\-]*(?:and|&)?[\s\-]*cost[\s\-]*based|weightage\s+(?:on|of|to|for)\s+(?:the\s+)?techn",
    re.I)
NEGATION = re.compile(r"\b(?:not|no|nor|without)\b[^.]{0,30}$", re.I)
L1_RE = re.compile(r"\bL-?1\b|lowest\s+(?:quote|bid|price|evaluated|rate)|reverse auction", re.I)


def detect_mode(text):
    for m in QCBS_RE.finditer(text):
        if not NEGATION.search(text[max(0, m.start() - 40):m.start()]):
            return "QCBS"
    return "L1" if L1_RE.search(text) else ""


def decide_mode(model_value, clean):
    """QCBS wording in the text always wins. Otherwise use the model's answer.
    If neither says anything, report UNKNOWN instead of guessing L1."""
    rule, llm, warns = detect_mode(clean), (model_value or "").strip().upper(), []
    if rule == "QCBS":
        if llm == "L1":
            warns.append("tech_evaluation_mode: model said L1 but the text mentions QCBS; used QCBS")
        return "QCBS", warns
    if llm in ("L1", "QCBS"):
        if llm == "QCBS":
            warns.append("tech_evaluation_mode: model said QCBS but the word QCBS / weightage is not in the text; please check")
        return llm, warns
    if rule:
        return rule, warns
    return "UNKNOWN", ["tech_evaluation_mode: evaluation method not found in the tender"]

# ----------------------------------------------------------------------------
# Checks on the answers (warnings only; the six output keys never change)
# ----------------------------------------------------------------------------

def alnum(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def validate(res, clean_text):
    warns, src = [], alnum(clean_text)
    for k in ("name of work", "ato_criteria", "similar_work_definition"):
        v = res[k]
        if not v:
            warns.append(f"{k}: empty")
        elif alnum(v) not in src:
            warns.append(f"{k}: not found word-for-word in the source (possible paraphrase)")
    if res["ato_criteria"] and not AMOUNT_RE.search(res["ato_criteria"]):
        warns.append("ato_criteria: no rupee amount found in the answer")
    if re.search(r"not less than", res["similar_work_definition"], re.I):
        warns.append("similar_work_definition: may contain money thresholds that belong to the experience criteria")
    if not res["documents_to_be_submitted"]:
        warns.append("documents_to_be_submitted: empty")
    for item in res["documents_to_be_submitted"]:
        if alnum(item) not in src:
            warns.append(f"documents_to_be_submitted: item not word-for-word in source -> {item[:60]}")
    if not res["tender no"]:
        warns.append("tender no: pattern found nothing")
    return warns


MARKER = re.compile(r"^[\s\u2022\u27a2>*\-\u2013]+|^\(?(?:[ivx]+|\d+|[a-z])[.)]\s+", re.I)


def clean_item(s):
    return MARKER.sub("", s.strip()).strip()


def weak_fields(result, clean):
    """Fields that probably need a second try with a wider search."""
    src, weak = alnum(clean), []
    for snake, key in (("name_of_work", "name of work"), ("ato_criteria", "ato_criteria"),
                       ("similar_work_definition", "similar_work_definition")):
        v = result[key]
        if not v or alnum(v) not in src:
            weak.append(snake)
    if "name_of_work" not in weak and len(result["name of work"]) < 8:
        weak.append("name_of_work")
    if "ato_criteria" not in weak and not AMOUNT_RE.search(result["ato_criteria"]):
        weak.append("ato_criteria")
    if len(result["documents_to_be_submitted"]) < 3:
        weak.append("documents_to_be_submitted")
    return weak

# ----------------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------------

FIELD_KEYS = {
    "name_of_work": "name of work",
    "ato_criteria": "ato_criteria",
    "similar_work_definition": "similar_work_definition",
    "documents_to_be_submitted": "documents_to_be_submitted",
}


def to_result(d, tender_no):
    return {
        "name of work": (d.get("name_of_work") or "").strip(),
        "tender no": tender_no,
        "ato_criteria": (d.get("ato_criteria") or "").strip(),
        "similar_work_definition": (d.get("similar_work_definition") or "").strip(),
        "documents_to_be_submitted": [x for x in (clean_item(i) for i in d.get("documents_to_be_submitted") or []) if x],
        "tech_evaluation_mode": (d.get("tech_evaluation_mode") or "").strip().upper(),
    }


def build_context(scale, limit, elements, only=None):
    """Pick the passages for the model, each passage once. If the text is too long
    for the model's window, search a little narrower until it fits."""
    while True:
        _, ids, chunks = retrieve_all(elements, scale=scale, only=only)
        text = merge_context(ids, chunks)
        if len(text) <= limit or scale <= 0.4:
            return text
        scale *= 0.85


def _iter_tables(elements):
    for element in elements:
        if element["type"] != "table":
            continue
        yield element
        for row in element["rows"]:
            for cell in row:
                if isinstance(cell, dict):
                    yield from _iter_tables(cell.get("elements", []))


def save_parse_artifacts(outdir, stem, raw, clean, elements):
    """Save parsed layout, chunks, and extracted tables as separate JSON files."""
    parsed_path = outdir / f"{stem}.parsed_response.json"
    chunks_path = outdir / f"{stem}.chunks.json"
    tables_path = outdir / f"{stem}.tables.json"

    parsed_path.write_text(json.dumps({
        "raw_text": raw,
        "clean_text": clean,
        "elements": elements,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    chunks = chunk_elements(elements)
    chunks_path.write_text(json.dumps([
        {"index": index, "character_count": len(text), "text": text}
        for index, text in enumerate(chunks)
    ], indent=2, ensure_ascii=False), encoding="utf-8")

    tables = []
    for index, element in enumerate(_iter_tables(elements)):
        rows = element["rows"]
        if rows and rows[0] and isinstance(rows[0][0], dict):
            rows = [[cell["text"] for cell in row] for row in rows]
        table = {"index": index, "rows": rows}
        for key in ("page", "bbox"):
            if key in element:
                table[key] = element[key]
        tables.append(table)
    tables_path.write_text(json.dumps(tables, indent=2, ensure_ascii=False), encoding="utf-8")


def process(path, args, system_prompt):
    stem = Path(path).stem
    outdir = Path(args.out)
    artifact_dir = Path(args.artifacts_out)
    outdir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and (outdir / f"{stem}.json").exists():
        print(f"[{stem}] already done, skipped")
        return

    raw, clean, elements = load_document(path)
    save_parse_artifacts(artifact_dir, stem, raw, clean, elements)
    tender_no = get_tender_no(raw)

    limit = args.max_context_chars or 60000

    user = (
        "TENDER TEXT (passages from the tender; '...' marks skipped parts):\n\n"
        + build_context(args.scale, limit, elements=elements)
    )

    if args.dump_text or args.dry_run:
        (artifact_dir / f"{stem}.parsed.txt").write_text(clean, encoding="utf-8")
        (artifact_dir / f"{stem}.context.txt").write_text(user, encoding="utf-8")

    if args.dry_run:
        print(f"[{stem}] tender no = {tender_no!r} | rule-based mode = {detect_mode(clean) or 'not found'} "
              f"| parsed = {len(clean)} chars | sent = {len(user)} chars (limit {limit})")
        return

    result = to_result(ask_llm(system_prompt, user, args.model), tender_no)

    # second try, with a wider search, only for fields that look missed
    retried = []
    if args.retry:
        weak = weak_fields(result, clean)
        if weak:
            wide = build_context(
                args.scale * 2, limit, only=weak, elements=elements,
            )
            r2 = to_result(ask_llm(
                system_prompt,
                "TENDER TEXT (passages from the tender; '...' marks skipped parts):\n\n" + wide,
                args.model,
            ),
                           tender_no)
            for snake in weak:
                key = FIELD_KEYS[snake]
                if r2[key]:
                    result[key] = r2[key]
            retried = weak

    result["tech_evaluation_mode"], mode_warns = decide_mode(result["tech_evaluation_mode"], clean)
    warns = validate(result, clean) + mode_warns
    if retried:
        warns.append(f"second try with wider search for: {retried}")

    (outdir / f"{stem}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if warns:
        (artifact_dir / f"{stem}.warnings.json").write_text(json.dumps(warns, indent=2), encoding="utf-8")
    print(f"[{stem}] done -> {outdir / (stem + '.json')}" + (f"  ({len(warns)} warnings)" if warns else ""))
    for w in warns:
        print("   !", w, file=sys.stderr)


def main():
    load_dotenv(HERE / ".env")
    ap = argparse.ArgumentParser(description="Extract tender fields from PDF/DOCX/DOC into JSON")
    ap.add_argument("files", nargs="+")
    ap.add_argument(
        "--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
        help="Gemini model ID (defaults to GEMINI_MODEL from .env)",
    )
    ap.add_argument("--max-context-chars", type=int, default=0, help="override how much text is sent to the model")
    ap.add_argument("--scale", type=float, default=1.0, help="hybrid: search wider (>1) or narrower (<1)")
    ap.add_argument("--no-retry", dest="retry", action="store_false", help="turn off the second try for weak fields")
    ap.add_argument("--workers", type=int, default=1, help="files processed at the same time")
    ap.add_argument("--skip-existing", action="store_true", help="skip files that already have an output JSON")
    ap.add_argument("--out", default="results", help="folder for final extraction JSON files")
    ap.add_argument("--artifacts-out", default="output", help="folder for parsed text, chunks, tables, and warnings")
    ap.add_argument("--dump-text", action="store_true", help="save the parsed text and the text sent to the model")
    ap.add_argument("--dry-run", action="store_true", help="read files and find tender no only, no model call")
    args = ap.parse_args()

    system_prompt = PROMPT_FILE.read_text(encoding="utf-8")

    def run(f):
        try:
            process(f, args, system_prompt)
        except Exception as e:
            print(f"[{Path(f).stem}] FAILED: {e}", file=sys.stderr)

    if args.workers > 1:
        with ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(run, args.files))
    else:
        for f in args.files:
            run(f)


if __name__ == "__main__":
    main()
