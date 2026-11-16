#!/usr/bin/env python3
"""
Checks, WITHOUT calling any model, whether the right text reaches the model.

1. Retrieval check: for each field, "anchors" are short phrases that must be in the
   text sent to the model, or the answer cannot be extracted. Score = found / total.
   This now tests the text the model really receives (all fields merged, each
   passage once), and compares it with the old keyword-window method.
2. Tender number and L1/QCBS check against known answers.

Add your own files to GOLD / EXPECT. Use files the code has never been tuned on:
those are the ones that tell you whether it works in general.

    python eval_recall.py "samples/Sample 1.docx" "samples/Sample 2.pdf" "samples/Sample 3.docx"
"""
import argparse
import re
from pathlib import Path

from document_parser import load_document
from extract_tender import get_tender_no, detect_mode, build_context

# phrases that must be present in the text sent to the model
GOLD = {
    "Sample_1": {
        "name_of_work": [r"Procurement of 1\.0 & 1\.25 mm MS Plain", r"Epi coated Barrels for Chennai"],
        "ato_criteria": [r"Financial Criteria \(Annual Turnover\)", r"Annual Turnover PQC value",
                         r"any one of the preceding 3 years"],
        "similar_work_definition": [r"Definition of Similar Item \(Order\)", r"MS Composite Barrels/Drums"],
        "documents_to_be_submitted": [r"ANNEXURES: TO BE SUBMITTED BY ALL THE BIDDERS",
                                      r"Undertaking by the Tenderer\(s\) on acceptance",
                                      r"PAN Card", r"GST Registration certificate"],
        "tech_evaluation_mode": [r"selection of L1 Bidder will be based on Overall Lowest Gross Delivered Rate"],
    },
    "Sample_2": {
        "name_of_work": [r"Supply, Fabrication, Erection and Laying of pipelines including",
                         r"HDD and hookup with existing & operational Propane"],
        "ato_criteria": [r"17\.1 Annual Turnover Criteria", r"9,36,31,000"],
        "similar_work_definition": [r"17\.2\.1 Definition of similar work", r"irrespective of the order value"],
        "documents_to_be_submitted": [r"18\. Other Commercial documents to be submitted",
                                      r"Partnership deed or Certificate of Incorporation",
                                      r"Annexure-J of Part-B"],
        "tech_evaluation_mode": [r"Lowest quote available after Reverse Auction shall be treated as L-1"],
    },
    "Sample_3": {
        "name_of_work": [r"Development and operation of Loyalty Management Platform for Servo"],
        "ato_criteria": [r"shall be at least following values during any of the preceding three financial years"],
        "similar_work_definition": [r"DEFINITION OF SIMILAR WORKS", r"e-Wallet",
                                    r"excluded from the value of similar work orders"],
        "documents_to_be_submitted": [r"Undertaking by the tenderer\(s\) on Acceptance of tender terms",
                                      r"Proforma of Tender not Tampered",
                                      r"Compliance Certificate for bidders from a country"],
        "tech_evaluation_mode": [r"QCBS \(quality cost-based selection\) methodology", r"70% weightage"],
    },
}

KEYWORDS = re.compile(
    r"name of (?:the )?work|nature of work|subject|turnover|similar work|similar completed|"
    r"work experience|previous work order|definition of|documents?\s+(?:to be|required|for)|"
    r"to be submitted|checklist|evaluation|qcbs|quality and cost|\bL-?1\b|lowest|"
    r"technical (?:score|marks)|weightage|reverse auction",
    re.I,
)

# known answers: (tender no as written on the cover page, L1 or QCBS)
EXPECT = {
    "Sample_1": ("MnC/LUB-19/PT-50/26-27", "L1"),
    "Sample_2": ("HCC/ENG-29/PT-95/24-25", "L1"),
    "Sample_3": ("MnC/LUB-13/PT-33/26-27", "QCBS"),
}


def norm(s):
    return re.sub(r"\s+", " ", s)


def score(ctx, anchors):
    c = norm(ctx)
    return sum(1 for a in anchors if re.search(a, c, re.I)), len(anchors)


def select_chunks(text, before=3, after=30, head=40, max_chars=24000):
    """Keep the original keyword-window baseline for comparing retrieval quality."""
    lines = text.splitlines()
    keep = set(range(min(head, len(lines))))
    for index, line in enumerate(lines):
        if KEYWORDS.search(line):
            keep.update(range(max(0, index - before), min(len(lines), index + after + 1)))
    out, total, prev = [], 0, -2
    for index in sorted(keep):
        if index != prev + 1:
            out.append("...")
        out.append(lines[index])
        total += len(lines[index]) + 1
        prev = index
        if total >= max_chars:
            break
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    limit = 60000

    tot = {"old": [0, 0], "new": [0, 0]}
    side = [0, 0]
    for f in a.files:
        stem = Path(f).stem.replace(" ", "_")
        raw, clean, elements = load_document(f)

        if stem in EXPECT:
            want_no, want_mode = EXPECT[stem]
            got_no, got_mode = get_tender_no(raw), detect_mode(clean)
            for label, got, want in (("tender no", got_no, want_no), ("L1/QCBS", got_mode, want_mode)):
                ok = "OK " if got == want else "BAD"
                side[0] += got == want
                side[1] += 1
                print(f"[{stem}] {label:10s} {ok} got {got!r}  expected {want!r}")

        if stem not in GOLD:
            print(f"(no retrieval gold for {stem}, skipped)")
            continue
        old_ctx = select_chunks(clean, max_chars=24000)
        new_ctx = build_context(1.0, limit, elements=elements)

        print(f"== {stem}   old method sends {len(old_ctx)} chars   new method sends {len(new_ctx)} chars")
        print(f"{'field':28s} {'old':>6s} {'new':>6s}   missing in new")
        for field, anchors in GOLD[stem].items():
            o, n = score(old_ctx, anchors), score(new_ctx, anchors)
            miss = [x for x in anchors if not re.search(x, norm(new_ctx), re.I)]
            print(f"{field:28s} {o[0]}/{o[1]:<4d} {n[0]}/{n[1]:<4d}   {miss if miss else ''}")
            tot["old"][0] += o[0]; tot["old"][1] += o[1]
            tot["new"][0] += n[0]; tot["new"][1] += n[1]
        print()
    print(f"OVERALL retrieval  old {tot['old'][0]}/{tot['old'][1]} = {tot['old'][0]/max(1,tot['old'][1]):.0%}"
          f"   new {tot['new'][0]}/{tot['new'][1]} = {tot['new'][0]/max(1,tot['new'][1]):.0%}")
    print(f"Tender no + L1/QCBS checks passed: {side[0]}/{side[1]}")


if __name__ == "__main__":
    main()
