"""
Text Preprocessing Module
==========================
Handles complete text cleaning pipeline:
  - Unicode-safe encoding
  - Lowercase conversion
  - URL removal
  - HTML entity decoding and tag stripping
  - Duplicate/extra whitespace collapsing
  - Punctuation preservation
"""

import re
import html
import logging
from typing import List
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns (compiled once at import time for speed)
# ---------------------------------------------------------------------------
_URL_PATTERN = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+!*(),]|(?:%[0-9a-fA-F]{2}))+"
)
_HTML_TAG_PATTERN = re.compile(r"<[^<>]+?>")
_EXTRA_SPACE_PATTERN = re.compile(r"[ \t]+")      # collapse horizontal whitespace only
_NEWLINE_PATTERN = re.compile(r"\n{3,}")           # collapse 3+ newlines to 2


class TextPreprocessor:
    """
    Stateless text preprocessing class.

    Applies a fixed sequence of cleaning operations to produce normalised text
    suitable for both TF-IDF vectorisation and handcrafted feature extraction.

    Operations (in order):
      1. Coerce to string
      2. Unicode-safe re-encoding (drop malformed bytes)
      3. HTML entity decoding  (&amp; → &, etc.)
      4. HTML tag removal      (<b>bold</b> → bold)
      5. URL removal
      6. Lowercase conversion
      7. Collapse horizontal whitespace (tabs/spaces → single space)
      8. Collapse excessive newlines (3+ → 2)
      9. Strip leading/trailing whitespace
    """

    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.

        Args:
            text: Raw input string (any type is handled safely).

        Returns:
            Cleaned, normalised string.  Returns "" for None / un-parsable input.
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        try:
            # 1. Unicode safety — drop undecodable bytes
            text = text.encode("utf-8", errors="ignore").decode("utf-8")

            # 2. Decode HTML entities before tag removal so we don't lose content
            text = html.unescape(text)

            # 3. Strip HTML tags
            text = _HTML_TAG_PATTERN.sub(" ", text)

            # 4. Remove URLs
            text = _URL_PATTERN.sub(" ", text)

            # 5. Lowercase
            text = text.lower()

            # 6. Collapse horizontal whitespace (preserve newlines for sentence count)
            text = _EXTRA_SPACE_PATTERN.sub(" ", text)

            # 7. Collapse excessive newlines
            text = _NEWLINE_PATTERN.sub("\n\n", text)

            # 8. Strip edges
            text = text.strip()

            return text

        except Exception as exc:  # pragma: no cover
            logger.warning("Error cleaning text: %s", exc)
            return ""

    def clean_texts(
        self, texts: List[str], show_progress: bool = True
    ) -> List[str]:
        """
        Clean a list of texts, optionally showing a tqdm progress bar.

        Args:
            texts:         Iterable of raw strings.
            show_progress: Display tqdm progress bar when True.

        Returns:
            List of cleaned strings (same length as *texts*).
        """
        iterable = (
            tqdm(texts, desc="Cleaning texts", unit="doc", dynamic_ncols=True)
            if show_progress
            else texts
        )
        return [self.clean_text(t) for t in iterable]

    def tokenize(self, text: str) -> List[str]:
        """
        Whitespace-tokenise text (preserves punctuation tokens).

        Args:
            text: Input string.

        Returns:
            List of whitespace-split tokens.
        """
        if not isinstance(text, str):
            return []
        return text.split()


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """Preprocess a single text string (convenience wrapper)."""
    return TextPreprocessor().clean_text(text)


def preprocess_texts(texts: List[str], show_progress: bool = True) -> List[str]:
    """Preprocess a list of text strings (convenience wrapper)."""
    return TextPreprocessor().clean_texts(texts, show_progress=show_progress)
