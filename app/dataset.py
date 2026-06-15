import re
import hashlib
from collections import Counter
from datasets import load_dataset

# ─────────────────────────────────────────────────────────────────────
# Multi-Stage Preprocessing Pipeline
#
# The ArXiv ML Papers corpus contains academic abstracts. While
# inherently cleaner than web scrapes, the text still contains structural noise:
#
#   - URLs (http://, ftp://) that bias embeddings toward web junk
#   - Email addresses of authors
#   - LaTeX artifacts or file paths from formatting
#   - Excessive whitespace from PDF parsing
#   - Cross-posted or updated paper duplicates
#
# Each preprocessing stage below targets a specific noise source.
# The comments explain *why* each step matters for downstream
# embedding quality and clustering accuracy.
# ─────────────────────────────────────────────────────────────────────

# Compiled regex patterns — defined once, reused across all documents.
# Pre-compilation avoids re-parsing the regex on every document.

# URLs — matches http://, https://, ftp:// and www. prefixed links.
# These are structural artifacts of email communication, not topical content.
# An embedding model would learn spurious associations between URLs and topics.
_URL_PATTERN = re.compile(r'https?://\S+|ftp://\S+|www\.\S+', re.IGNORECASE)

# Email addresses — encode sender/recipient identity, not discussion topic.
# A post about gun control should not embed closer to other posts by the
# same author just because they share an email address.
_EMAIL_PATTERN = re.compile(r'\S+@\S+\.\S+')

# File paths or LaTeX artifacts.
# They carry no semantic meaning for topic
# classification and would create false similarity between unrelated papers
# that happen to mention the same directory structure in a methodology section.
_FILEPATH_PATTERN = re.compile(r'(?:[A-Za-z]:\\[\w\\.-]+|/(?:usr|etc|var|home|tmp|bin|lib|opt|dev)[\w/.-]*)')

# Whitespace normalisation — collapse runs of spaces/tabs/newlines.
# Parses from PDFs can leave 5-10 blank lines in a row.
# These waste embedding capacity on nothing.
_WHITESPACE_PATTERN = re.compile(r'[ \t\r\f\v]+')

# Non-printable / control characters — binary junk from encoding issues.
# Some papers contain garbled characters from charset mismatches.
_CONTROL_CHARS_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')


def _clean_text(text: str) -> str:
    """
    Apply regex-based cleaning to a single document.

    Order matters:
    1. Remove URLs first (they contain @ and / which would partially
       match email and filepath patterns if processed later)
    2. Remove emails
    3. Remove file paths
    4. Strip control characters
    5. Normalise whitespace last (after removing items that create gaps)

    Returns:
        Cleaned text string, stripped of structural noise.
    """
    text = _URL_PATTERN.sub(' ', text)
    text = _EMAIL_PATTERN.sub(' ', text)
    text = _FILEPATH_PATTERN.sub(' ', text)
    text = _CONTROL_CHARS_PATTERN.sub('', text)
    text = _WHITESPACE_PATTERN.sub(' ', text)
    return text.strip()


def _compute_fingerprint(text: str) -> str:
    """
    Compute a content fingerprint for near-duplicate detection.

    Strategy: extract character trigrams, sort them, hash the result.
    This is a lightweight alternative to full MinHash/LSH. It catches:
    - Exact duplicates (same paper published in multiple tracks)
    - Near-duplicates (same paper with minor abstract updates)

    Why trigrams, not full text hash?
    A full MD5 of the raw text would miss near-duplicates — a single
    extra space would produce a completely different hash. Trigram-based
    fingerprinting is robust to minor whitespace/punctuation differences.

    I normalise to lowercase and strip non-alphanumeric characters first,
    so "Hello World!" and "hello world" produce the same fingerprint.
    """
    # Normalise: lowercase, keep only alphanumeric + spaces
    normalised = re.sub(r'[^a-z0-9 ]', '', text.lower())
    # Extract sorted character trigrams
    trigrams = sorted(set(
        normalised[i:i+3] for i in range(len(normalised) - 2)
    ))
    # Hash the trigram set — deterministic fingerprint
    fingerprint = hashlib.md5(''.join(trigrams).encode()).hexdigest()
    return fingerprint


def load_documents():
    """
    Load and preprocess the ArXiv ML Papers dataset.

    Preprocessing decisions:
    ─────────────────────────────────────────────────────────────────────
    1. Dataset: CShorten/ML-ArXiv-Papers
       - Contains ~117k ML papers with titles and abstracts.
       - Clean text without HTML or quotes.

    2. documents[:5000]
       - We cap at 5000 to allow rapid experimentation while preserving topic
         diversity across categories.

    3. Label Mapping
       - Extracts primary ArXiv category (e.g. 'cs.LG') from 'categories'.
       - Maps to integer labels for GMM clustering.

    Pipeline stages:
       Stage 1 → Regex cleaning (URLs, emails, file paths, whitespace)
       Stage 2 → Length filtering (remove sub-50-char documents)
       Stage 3 → Near-duplicate removal (trigram fingerprinting)
    ─────────────────────────────────────────────────────────────────────

    Returns:
        documents:   list of cleaned text strings
        labels:      list of integer category labels
        label_names: list of category name strings
    """

    print("Downloading/Loading ArXiv dataset from HuggingFace...")
    dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train")

    raw_documents = []
    raw_category_strings = []
    
    for item in dataset.select(range(6000)):
        title = item.get("title", "").strip()
        abstract = item.get("abstract", "").strip()
        
        # Combine title and abstract
        doc_text = f"{title}\n\n{abstract}"
        raw_documents.append(doc_text)
        
        # Since the HuggingFace dataset 'CShorten/ML-ArXiv-Papers' lacks the 'categories' column,
        # we assign a synthetic category based on simple keyword heuristics for evaluation purposes.
        lower_text = doc_text.lower()
        if any(kw in lower_text for kw in ["vision", "image", "pixel", "convolutional", "cnn"]):
            primary_cat = "cs.CV"
        elif any(kw in lower_text for kw in ["language", "text", "nlp", "translation", "bert", "transformer"]):
            primary_cat = "cs.CL"
        elif any(kw in lower_text for kw in ["reinforcement", "reward", "agent", "mdp"]):
            primary_cat = "cs.LG.RL"
        elif any(kw in lower_text for kw in ["graph", "node", "edge", "gnn"]):
            primary_cat = "cs.SI"
        elif any(kw in lower_text for kw in ["time series", "forecasting", "temporal"]):
            primary_cat = "stat.AP"
        else:
            primary_cat = "cs.LG"
            
        raw_category_strings.append(primary_cat)

    # Build category mappings
    unique_cats = sorted(list(set(raw_category_strings)))
    cat_to_id = {cat: idx for idx, cat in enumerate(unique_cats)}
    label_names = unique_cats
    
    raw_labels = [cat_to_id[c] for c in raw_category_strings]

    # Limit to 5000 before pipeline
    raw_documents = raw_documents[:5000]
    raw_labels = raw_labels[:5000]

    total_raw = len(raw_documents)

    # ── Stage 1: Regex cleaning ──
    # Apply multi-pattern cleaning to strip structural noise.
    # This runs before length filtering because URL/email removal
    # can reduce a document's meaningful content below the threshold.
    cleaned_pairs = [
        (_clean_text(doc), label)
        for doc, label in zip(raw_documents, raw_labels)
    ]

    # ── Stage 2: Length filtering ──
    # After stripping noise, some posts become empty or near-empty.
    # Sub-50-char documents produce meaningless embeddings that sit
    # near the origin in embedding space, polluting clusters.
    MIN_DOC_LENGTH = 50
    length_filtered = [
        (doc, label) for doc, label in cleaned_pairs
        if len(doc.strip()) > MIN_DOC_LENGTH
    ]
    removed_by_length = len(cleaned_pairs) - len(length_filtered)

    # ── Stage 3: Near-duplicate removal ──
    # ArXiv papers can have updated versions or cross-listings that
    # appear as duplicates. Duplicates artificially inflate certain cluster
    # sizes and bias the GMM toward over-representing popular papers
    # rather than broad topics.
    seen_fingerprints = set()
    deduplicated = []
    for doc, label in length_filtered:
        fp = _compute_fingerprint(doc)
        if fp not in seen_fingerprints:
            seen_fingerprints.add(fp)
            deduplicated.append((doc, label))
    removed_by_dedup = len(length_filtered) - len(deduplicated)

    documents, labels = zip(*deduplicated) if deduplicated else ([], [])

    # ── Preprocessing stats ──
    # Transparency: show exactly what the pipeline did so the reader
    # can evaluate whether the filtering is too aggressive or too lenient.
    print(f"\n{'-'*55}")
    print("\n[ PREPROCESSING PIPELINE STATS ]")
    print(f"{'-'*55}")
    print(f"   Raw documents loaded:      {total_raw}")
    print(f"   After regex cleaning:       {len(cleaned_pairs)} (cleaned in-place)")
    print(f"   Removed by length (<{MIN_DOC_LENGTH} chars): {removed_by_length}")
    print(f"   Removed as duplicates:      {removed_by_dedup}")
    print(f"   Final corpus size:          {len(documents)}")
    print(f"   Categories:                 {len(label_names)}")
    print(f"{'-'*55}")

    # Category distribution — check that I haven't accidentally
    # wiped out entire categories during filtering.
    cat_counts = Counter(labels)
    print("   Category distribution:")
    for cat_id, count in sorted(cat_counts.items()):
        print(f"     {label_names[cat_id][:30]:<30s} {count:>4d} docs")
    print(f"{'-'*55}")

    print(f"\n[PREPROCESSING COMPLETE] Loaded {len(documents)} documents after preprocessing.")
    print(f"   Example doc preview: {documents[0][:100]!r}")

    return list(documents), list(labels), label_names
