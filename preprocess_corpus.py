#!/usr/bin/env python3
"""
pipeline_2.py -- Corpus Segmentation and Lexical Normalization
================================================================================

Stage 0 of the LexiLaw-NLP-Pipeline: converts raw legislative/regulatory text
files into the segmented, lexically-normalized `*_LIMPOS.txt` corpus consumed
by the main analysis notebook (`LexiLaw_NLP_PIPELINE_clean.ipynb`).

This script implements the cleaning methodology described in the manuscript's
Section 3 (Materials and Methods):

  1. Segmentation by legal device (Article/Section, regex-based), with a
     paragraph-level fallback for documents that contain too few formal
     delimiters to segment reliably.
  2. Strict-precedence substitution of multi-word institutional/technical
     terms into single unified tokens (CONCEPT_MAP), so that downstream
     tokenizers and stopword filters do not fragment or discard concepts
     such as "high-risk AI system".
  3. Standard English stopword removal (NLTK) plus a custom stoplist for
     legislative boilerplate and institutional noise.

Usage
-----
As a script, against a directory of raw `.txt` files (no Colab required --
this is the intended path for anyone running the pipeline outside Google
Colab, e.g. cloning this repository):

    python pipeline_2.py --input-dir raw_texts/ --output-dir corpus/

Each `document.txt` becomes `corpus/document_LIMPOS.txt`, matching the
naming convention `Stage 0` of the main notebook expects (`*_LIMPOS.txt` or
`* LIMPOS.txt`).

In Google Colab, run the script directly (or paste its contents into a
cell). It detects the Colab runtime automatically and falls back to the
original interactive `files.upload()` / `files.download()` workflow instead
of the CLI.

As a library:

    from pipeline_2 import segment_document, clean_and_tokenize_segment

    segments = segment_document(raw_text)
    tokens = clean_and_tokenize_segment(segments[0])

Scope notes (read before treating this as identical to the main notebook)
---------------------------------------------------------------------------
This script is a separate, earlier stage of the pipeline than the main
analysis notebook, and the two were not written to share code. Two
differences are worth knowing about rather than discovering by surprise:

* The legal-device regex here (`Art.`/`Article`/`Section`/`Sec.` followed by
  a number, with a fallback below 5 matches) is narrower than the main
  notebook's own `LEGAL_DEVICE_SPLIT_REGEX` (which also matches
  `CHAPTER`/`CAPÍTULO`/`§`, anchors at line start, and falls back below 15
  matches). Both are preserved as-is; do not assume they produce the same
  segment counts on the same raw text.
* This script's stopword filtering is a single flat pass (NLTK English
  stopwords + `CUSTOM_STOPWORDS` below) with no lemmatization or POS
  tagging. The main notebook applies its own, separate lemmatized
  preprocessing on top of this script's output for topic modeling. This
  script's job is only to produce the base `_LIMPOS.txt` corpus -- it is
  not meant to be the final normalization pass.
* `CUSTOM_STOPWORDS` includes a handful of entries specific to one document
  in this corpus (`psd`, `mg`, `rodrigo`, `pacheco`) -- these are signature
  -block artifacts (party/state abbreviation and the name of the Brazilian
  Senate President who signed the BR_2025 substitute), not general-purpose
  legal stopwords. They are harmless for the other five documents but make
  this stoplist corpus-specific rather than portable to an unrelated
  legislative corpus; flagged here so it isn't mistaken for a general-domain
  list if reused.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

import nltk
from nltk.corpus import stopwords

# ------------------------------------------------------------------------------
# 1. Corpus-specific dictionaries (methodology, manuscript Section 3)
# ------------------------------------------------------------------------------

# CONCEPT_MAP: strict-precedence substitution of multi-word technical terms
# into single unified tokens, applied before stopword filtering. This keeps
# "high-risk AI system" and similar compound terms from being fragmented or
# partially discarded by the tokenizer/stopword pass below.
CONCEPT_MAP: Dict[str, str] = {
    r"\bhigh[- ]risk ai systems?\b": "high_risk_system",
    r"\bhigh[- ]risk artificial intelligence systems?\b": "high_risk_system",
    r"\bgeneral[- ]purpose ai models?\b": "general_purpose_ai_model",
    r"\bgeneral[- ]purpose ai systems?\b": "general_purpose_ai_system",
    r"\bgeneral[- ]purpose artificial intelligence systems?\b": "general_purpose_ai_system",
    r"\bartificial intelligence systems?\b": "artificial_intelligence_system",
    r"\balgorithmic impact assessments?\b": "algorithmic_impact_assessment",
    r"\bregulatory sandboxes?\b": "regulatory_sandbox",
}

# CUSTOM_STOPWORDS: legislative formatting, procedural, and institutional
# noise -- see the "Scope notes" section above regarding the last four
# entries, which are specific to one document's signature block.
CUSTOM_STOPWORDS: Set[str] = {
    "article", "provision", "system", "use", "secretary", "parliament",
    "shall", "may", "hereby", "thereof", "within", "upon", "pursuant",
    "art", "chapter", "section", "law", "act", "bill", "brazil", "european",
    "union", "united", "states", "psd", "mg", "rodrigo", "pacheco",
}

# Legal-device segmentation pattern: "Art. 5", "Article 5", "Section 12",
# "Sec. 3", case-insensitive. See the module docstring for how this differs
# from the main notebook's own segmentation regex.
LEGAL_DEVICE_PATTERN = re.compile(r"\b(?:art\.?|article|section|sec\.?)\s+\d+", re.IGNORECASE)

DEFAULT_OUTPUT_SUFFIX = "_LIMPOS.txt"
MIN_LEGAL_DEVICE_MATCHES = 5  # below this, fall back to paragraph splitting


def ensure_nltk_stopwords() -> None:
    """Download the NLTK stopwords corpus if it isn't already present locally."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def _standard_stopwords() -> Set[str]:
    """English stopwords from NLTK, cached at module scope after first call."""
    ensure_nltk_stopwords()
    return set(stopwords.words("english"))


# Computed once (not on every segment) -- a pure efficiency refactor of the
# original per-call `set(stopwords.words("english"))`; the filtering
# behavior is unchanged.
_FULL_STOPLIST: Set[str] = set()


def _get_full_stoplist() -> Set[str]:
    global _FULL_STOPLIST
    if not _FULL_STOPLIST:
        _FULL_STOPLIST = _standard_stopwords() | CUSTOM_STOPWORDS
    return _FULL_STOPLIST


# ------------------------------------------------------------------------------
# 2. Core pipeline functions
# ------------------------------------------------------------------------------

def segment_document(full_text: str, min_legal_units: int = MIN_LEGAL_DEVICE_MATCHES) -> List[str]:
    """Split a raw document into segments delimited by legal devices.

    Locates every "Article N" / "Section N" occurrence (regex, case
    -insensitive) and cuts the text between consecutive matches. If fewer
    than `min_legal_units` matches are found -- e.g. an executive order with
    no numbered articles -- falls back to splitting on newlines, keeping
    lines longer than 10 characters.

    Returns a list of raw (uncleaned) text segments.
    """
    matches = list(LEGAL_DEVICE_PATTERN.finditer(full_text))

    if len(matches) < min_legal_units:
        lines = full_text.split("\n")
        segments = [line.strip() for line in lines if len(line.strip()) > 10]
        print(f"  -> Fallback: splitting by paragraph (total: {len(segments)} segments)")
        return segments

    segments = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        segments.append(full_text[start:end].strip())
    print(f"  -> Legal-device segmentation (total: {len(segments)} segments)")
    return segments


def clean_and_tokenize_segment(raw_segment: str, stoplist: Set[str] | None = None) -> List[str]:
    """Clean and tokenize a single segment: lowercase, CONCEPT_MAP substitution,
    strip punctuation and standalone digits, then filter stopwords.

    `stoplist` defaults to NLTK English stopwords unioned with
    `CUSTOM_STOPWORDS`; pass a different set to reuse this function outside
    this corpus's specific noise list.
    """
    if stoplist is None:
        stoplist = _get_full_stoplist()

    text = raw_segment.lower()

    # Concept-term unification (must run before punctuation stripping, since
    # the replacement tokens are joined with underscores).
    for pattern, token in CONCEPT_MAP.items():
        text = re.sub(pattern, token, text)

    text = re.sub(r"[^\w\s]", " ", text)  # strip punctuation, keep underscores
    text = re.sub(r"\b\d+\b", " ", text)  # drop standalone numbers

    raw_tokens = text.split()
    return [
        token for token in raw_tokens
        if token not in stoplist and len(token) > 2
    ]


def process_text(full_text: str, label: str = "") -> List[str]:
    """Run the full segmentation + cleaning pass over one document's raw text.

    Returns the processed segments as space-joined token strings, one per
    non-empty segment -- this is the exact line format written to each
    `*_LIMPOS.txt` output file.
    """
    raw_segments = segment_document(full_text)
    processed_segments: List[str] = []
    for segment in raw_segments:
        tokens = clean_and_tokenize_segment(segment)
        if tokens:
            processed_segments.append(" ".join(tokens))
    if label:
        print(f"  -> {label}: {len(processed_segments)} non-empty segments after cleaning")
    return processed_segments


# ------------------------------------------------------------------------------
# 3a. CLI entry point (no Colab dependency -- the recommended path on GitHub)
# ------------------------------------------------------------------------------

def process_file(input_path: Path, output_dir: Path, suffix: str = DEFAULT_OUTPUT_SUFFIX) -> Path:
    """Process one raw `.txt` file and write its cleaned corpus file. Returns the output path."""
    print(f"\nProcessing: {input_path.name}")
    full_text = input_path.read_text(encoding="utf-8", errors="ignore")
    processed_segments = process_text(full_text, label=input_path.name)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (input_path.stem + suffix)
    output_path.write_text("\n".join(processed_segments), encoding="utf-8")
    print(f"  -> Written: {output_path}")
    return output_path


def run_cli(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Segment and lexically normalize raw legislative text files "
                     "into the *_LIMPOS.txt corpus format used by the main pipeline notebook."
    )
    parser.add_argument("--input-dir", required=True, type=Path,
                         help="Directory containing raw *.txt files.")
    parser.add_argument("--output-dir", required=True, type=Path,
                         help="Directory to write cleaned *_LIMPOS.txt files to.")
    parser.add_argument("--output-suffix", default=DEFAULT_OUTPUT_SUFFIX,
                         help=f"Filename suffix for cleaned output files (default: {DEFAULT_OUTPUT_SUFFIX!r}). "
                              "Note: the original draft of this script used '_LIMPO_SAIDA.txt', which does not "
                              "match the '*_LIMPOS.txt' convention the main notebook looks for -- the default "
                              "here has been corrected to that convention.")
    args = parser.parse_args(argv)

    input_files = sorted(args.input_dir.glob("*.txt"))
    if not input_files:
        print(f"No .txt files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    for path in input_files:
        process_file(path, args.output_dir, suffix=args.output_suffix)

    print(f"\nDone: {len(input_files)} file(s) written to {args.output_dir}")


# ------------------------------------------------------------------------------
# 3b. Interactive Colab workflow (upload / process / download), unchanged in
#     spirit from the original script -- used only when running inside Colab.
# ------------------------------------------------------------------------------

def _running_in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def run_colab_interactive() -> None:
    from google.colab import files  # available only inside Colab

    print("Select and upload the raw .txt files you want to clean:")
    uploaded_files = files.upload()

    for filename, content in uploaded_files.items():
        print(f"\nProcessing: {filename}")
        full_text = content.decode("utf-8")
        processed_segments = process_text(full_text)

        if not processed_segments:
            print("  Warning: no valid text segments remained after cleaning.")
            continue

        print(f"  -> Example (first 3 processed segments): {processed_segments[:3]}")

        output_filename = filename.replace(".txt", DEFAULT_OUTPUT_SUFFIX)
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(processed_segments))

        print(f"  -> Downloading: {output_filename}")
        files.download(output_filename)


# ------------------------------------------------------------------------------
# 4. Entry point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    ensure_nltk_stopwords()
    if _running_in_colab():
        run_colab_interactive()
    else:
        run_cli()
