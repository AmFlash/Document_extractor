## 📂 File-by-File Breakdown

### 1. `run_pipeline.py` (The Starter)
* **Role:** Acts as the entry point for batch processing[cite: 8].
* **What it does:** Scans the `samples/` directory for supported documents (`.pdf`, `.docx`, `.doc`), sets up output directories (`results/` and `output/`), and kicks off `extract_tender.py` as a subprocess with configurable worker threads[cite: 8].

### 2. `extract_tender.py` (The Main Manager)
* **Role:** Coordinates the end-to-end extraction logic for each file[cite: 4].
* **What it does:** 
  * Ingests documents and saves intermediate JSON artifacts (parsed layout, chunks, tables) into the output directory[cite: 4].
  * Extracts the tender reference number using robust regex patterns (`get_tender_no`) and evaluates the evaluation mode (`detect_mode` for L1 vs. QCBS) without relying on the LLM[cite: 4].
  * Combines the retriever and LLM call to build final outputs, performing optional secondary retries for weaker fields before writing out the final JSON[cite: 4].

### 3. `document_parser.py` (The File Reader)
* **Role:** Handles multi-format document parsing and text cleaning[cite: 2].
* **What it does:**
  * Uses `pdfplumber` for PDFs to extract layout elements, text lines, and tables while dynamically stripping out repeating headers/footers[cite: 2].
  * Parses paragraphs and nested tables for Word documents (`.docx`)[cite: 2].
  * Automatically converts legacy `.doc` files to `.docx` on the fly using headless LibreOffice if installed[cite: 2].

### 4. `chunking.py` (The Text Cutter)
* **Role:** Breaks unstructured text and layout elements into manageable pieces[cite: 1].
* **What it does:**
  * Flattens table rows into distinct readable units separated by boundary markers (`---`)[cite: 1].
  * Identifies strong headings (`STRONG`) to prevent splitting sentences or clauses awkwardly[cite: 1].
  * Controls chunk sizes using character caps (`min_chars`, `max_chars`) with small overlaps to maintain context continuity[cite: 1].

### 5. `retriever.py` (The Smart Searcher)
* **Role:** Filters out irrelevant document noise and retrieves context specifically tailored to each targeted field[cite: 7].
* **What it does:**
  * Computes BM25 keyword relevance scores across chunks using field-specific search queries[cite: 7].
  * Boosts chunks matching key structural headers and applies Reciprocal Rank Fusion (RRF) to select the best passages[cite: 7].
  * Stitches chosen chunks together in document order via `merge_context`, eliminating duplicates to keep the prompt concise[cite: 7].

### 6. `generation.py` & `extraction_prompt.txt` (The AI Layer)
* **Role:** Manages secure, schema-compliant communication with the Gemini model[cite: 5, 6].
* **What it does:**
  * `extraction_prompt.txt` defines strict system instructions enforcing verbatim extraction, strict table handling, and JSON-only output constraints[cite: 5].
  * `generation.py` structures the API request (`:generateContent`), enforces output schemas (`LLM_SCHEMA`), locks temperature to `0` for deterministic outputs, and manages automatic retries with exponential backoff[cite: 6].

---

## 🚀 Usage Guide

### 1. Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
