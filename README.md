

1. run_pipeline.py (The Starter)
This is the main starter file. It checks the samples/ folder for PDF, DOCX, or DOC files, and creates results/ and output/ folders if they aren't there. Then, it runs the extract_tender.py script to process all the files and makes sure the final JSON files get saved properly.

2. extract_tender.py (The Main)
This script manages the whole job for each document:

It reads the document, pulls out the text and tables, and saves them as backup files in the output folder.

It uses simple pattern matching to grab the tender number and figure out the evaluation style without even needing AI.

It searches for the important parts of the document, sends them to model, and checks the answers. If something looks missing or weak, it can do a wider search and try again before saving the final result.

3. document_parser.py (The File Reader)
This file opens up messy PDF, Word, or old .doc files and cleans them up:

For PDFs, it reads text and tables page by page and removes repeating headers and footers.

For Word files, it reads the paragraphs and tables. If it sees an old .doc file, it automatically uses LibreOffice in the background to convert it to .docx first.

It gives back the raw text, a clean text version, and a neat list of all the parts inside the document.

4. chunking.py (The Text Cutter)
Structure aware chunking is done.Since tender documents are massive, this file chops them up into smaller, bite-sized pieces so they are easier to search through:

It keeps table rows together and looks for headings so it doesn't cut sentences or paragraphs in half.

It keeps each piece under a certain size limit and makes sure they slightly overlap so no details get lost.

5. retriever.py (The Smart Searcher)
Instead of dumping the whole tender document into the AI, this script finds only the specific sections that matter for each question:

For reranking:

It uses keyword matching (called BM25) to look for things like "annual turnover" or "similar work".

It gives extra points to sections that match the right headings, then sorts and picks the best pieces.

Finally, it glues those chosen pieces together into one clean text in the right order, making sure there are no duplicate parts.

6. generation.py & extraction_prompt.txt (The AI Helper)
This handles talking to the Gemini AI:

extraction_prompt.txt is the instruction sheet for the AI. It tells it to extract the five required fields, copy the text word-for-word without rewriting it, ignore table lines, and output clean JSON only.

generation.py actually sends the request to the Gemini API, forces it to follow the exact JSON rules, keeps the creativity setting at zero so the answers stay factual, and automatically tries again if the network drops.

