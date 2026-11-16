"""
Finds the parts of a tender that matter for each field.

How it works (plain version)
  1. Cut the tender into small pieces (chunks) at clause / heading boundaries.
  2. For each field, score every chunk by keyword match (BM25) and label match
     (e.g. the words "Annual Turnover").
  3. Keep the best chunks plus the chunks right after them (lists and tables
     continue over several chunks).
  4. merge_context() joins the chunks chosen for ALL fields into one text,
     each chunk only once, so the model is not sent the same passage twice.

"""
import math
import re
from collections import Counter

from chunking import chunk_elements

# ----------------------------------------------------------------------------
# What to look for, per field
#   queries  : different ways tenders word the same thing
#   patterns : label patterns that suggest a chunk is relevant
#   strong   : patterns that are almost certainly the right heading
#   k        : how many top chunks to keep
#   budget   : max characters for this field
#   fwd      : how many following chunks to pull in (lists run on)
# ----------------------------------------------------------------------------

def _p(*pats):
    return [re.compile(p, re.I) for p in pats]


FIELDS = {
    "name_of_work": dict(
        queries=["name of work", "name of the work", "title of work", "name of the project",
                 "description of work", "subject of tender", "nature of work", "tender for"],
        patterns=_p(r"name of (the )?(work|project)", r"nature of work", r"title of (the )?work"),
        strong=_p(r"name of (the )?work"),
        head=5,                      # the title is usually in the opening table
        k=4, budget=6000),
    "ato_criteria": dict(
        queries=["annual turnover", "annual turnover criteria", "financial criteria annual turnover",
                 "minimum annual turnover preceding three financial years", "average annual turnover",
                 "turnover of the bidder standalone financials", "financial turnover eligibility"],
        patterns=_p(r"annual turnover", r"\bATO\b", r"financial criteria", r"\bturnover\b"),
        strong=_p(r"annual turnover (criteria|pqc)", r"\(annual turnover\)", r"^annual turnover$"),
        fwd=2, k=5, budget=8000),
    "similar_work_definition": dict(
        queries=["definition of similar work", "definition of similar item order",
                 "similar work shall mean", "similar supply definition", "similar nature of work",
                 "works involving laying", "bidder should have supplied similar item"],
        patterns=_p(r"definition of similar", r"similar (work|item|order|supply)s?\s+(shall|means?)",
                    r"similar (work|item)s? (criteria|experience)", r"similar nature"),
        strong=_p(r"definition of similar"),
        fwd=2, k=5, budget=8000),
    "documents_to_be_submitted": dict(
        queries=["documents to be submitted by the bidder", "other commercial documents to be submitted",
                 "list of documents required to be uploaded", "annexures to be submitted by all the bidders bid stage",
                 "declaration undertaking annexure part-b", "PAN GST registration power of attorney integrity pact",
                 "checklist of documents enclosures"],
        patterns=_p(r"documents?\W.{0,40}(to be )?(submitted|uploaded|required)",
                    r"annexures?\W.{0,30}to be submitted", r"\bPAN\b", r"integrity pact",
                    r"declaration|undertaking"),
        strong=_p(r"documents?\W.{0,40}to be (submitted|uploaded)", r"annexures?\W.{0,30}to be submitted",
                  r"list of documents"),
        fwd=3,                       # lists run across several chunks
        k=6, budget=12000),
    "tech_evaluation_mode": dict(
        queries=["tender evaluation criteria", "price bid evaluation criteria",
                 "selection of L1 bidder lowest", "QCBS quality cost based selection technical score weightage",
                 "reverse auction lowest quote L-1", "techno commercial evaluation",
                 "basis of award of contract"],
        patterns=_p(r"\bQCBS\b", r"evaluation criteria", r"\bL-?1\b", r"weightage",
                    r"technical (score|marks)", r"quality.{0,10}cost"),
        strong=_p(r"\bQCBS\b", r"(tender|price bid|techno.commercial) evaluation criteria", r"selection of L-?1"),
        fwd=2, k=6, budget=8000),
}

# ----------------------------------------------------------------------------
# BM25 keyword scoring
# ----------------------------------------------------------------------------

STOP = set("the of and to in a for be is by as on or an with shall at from this that are".split())


def tok(s):
    out = []
    for w in re.findall(r"[a-z0-9]+", s.lower()):
        if w in STOP:
            continue
        out.append(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out


class BM25:
    def __init__(self, docs, k1=1.4, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.dl = [len(d) for d in docs]
        self.avg = (sum(self.dl) / len(docs)) if docs else 1
        df = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, q):
        out = []
        for tf, dl in zip(self.tf, self.dl):
            s = 0.0
            for w in q:
                f = tf.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            out.append(s)
        return out

# ----------------------------------------------------------------------------
# Combining the scores and picking chunks
# ----------------------------------------------------------------------------

def _top(scores, n):
    return [i for i in sorted(range(len(scores)), key=lambda i: -scores[i])[:n] if scores[i] > 0]


def _rrf(rank_lists, weights, k=60):
    """Reciprocal rank fusion: a chunk ranked high in several lists wins."""
    sc = {}
    for ranks, w in zip(rank_lists, weights):
        for r, i in enumerate(ranks):
            sc[i] = sc.get(i, 0.0) + w / (k + r)
    return sorted(sc, key=sc.get, reverse=True)


def retrieve_all(elements, scale=1.0, only=None):
    """Return ({field: context}, {field: [chunk ids]}, chunks).
    scale > 1 widens the search (used for retry), scale < 1 shrinks it."""
    chunks = chunk_elements(elements)
    if not chunks:
        return {}, {}, chunks
    bm = BM25([tok(c) for c in chunks])
    ctx, ids = {}, {}
    for name, spec in FIELDS.items():
        if only and name not in only:
            continue
        lists, weights = [], []
        agg = [0.0] * len(chunks)
        for q in spec["queries"]:
            s = bm.scores(tok(q))
            agg = [a + b for a, b in zip(agg, s)]
            lists.append(_top(s, 25))
            weights.append(1.0)

        hits = []
        for i, c in enumerate(chunks):
            h = sum(1 for p in spec["patterns"] if p.search(c))
            if h:
                hits.append((h, agg[i], i))
        hits.sort(reverse=True)
        lists.append([i for _, _, i in hits[:25]])
        weights.append(2.0)                                  # label patterns count double

        strong = []
        for i, c in enumerate(chunks):
            h = sum(1 for p in spec.get("strong", []) if p.search(c))
            if h:
                strong.append((h, agg[i], i))
        strong.sort(reverse=True)
        if strong:
            lists.append([i for _, _, i in strong[:25]])
            weights.append(3.0)                              # exact heading match: strongest vote

        fused = _rrf(lists, weights)
        top = fused[: max(1, int(spec["k"] * scale))]
        head = [i for i in range(min(spec.get("head", 0), len(chunks))) if i not in top]
        top = head + top                                     # opening chunks (title block)

        budget = int(spec["budget"] * scale)
        fwd = spec.get("fwd", 1)
        order = list(top)                                    # hits first
        for r in range(1, fwd + 1):                          # then following chunks, nearest first
            order += [i + r for i in top]
        order += [i - 1 for i in top]                        # then the chunk before
        chosen, used = [], 0
        for i in order:
            if i < 0 or i >= len(chunks) or i in chosen:
                continue
            if used + len(chunks[i]) > budget and chosen:
                continue
            chosen.append(i)
            used += len(chunks[i])
        chosen.sort()

        parts, prev = [], None
        for i in chosen:
            if prev is not None and i != prev + 1:
                parts.append("...")
            parts.append(chunks[i])
            prev = i
        ctx[name], ids[name] = "\n".join(parts), chosen
    return ctx, ids, chunks


def merge_context(ids, chunks):
    """One text for all fields: every chosen chunk once, in document order."""
    chosen = sorted({i for v in ids.values() for i in v})
    parts, prev = [], None
    for i in chosen:
        if prev is not None and i != prev + 1:
            parts.append("...")
        parts.append(chunks[i])
        prev = i
    return "\n".join(parts)
