# ============================================================
# backend/preprocessor.py
# ============================================================
# PURPOSE: Clean and normalize raw text from resumes/job descriptions.
#
# WHY DO WE PREPROCESS?
# Raw text is messy: "Python!!!" and "python" mean the same thing,
# but a computer thinks they're different. Preprocessing makes text
# consistent so our ML model works better.
#
# STEPS WE APPLY:
#   1. Lowercase everything
#   2. Remove punctuation & special characters
#   3. Tokenize (split into words)
#   4. Remove stopwords ("the", "is", "and" — not useful for matching)
#   5. Lemmatize (reduce words to base form: "running" → "run")
# ============================================================

import re           # Regular expressions — for pattern-based text cleaning
import string       # Python's string constants (punctuation, etc.)
import nltk         # Natural Language Toolkit
import spacy        # Advanced NLP library

# ---- Download NLTK data (only needed once) ----
# NLTK needs to download its word lists the first time you use them
nltk.download("stopwords", quiet=True)    # List of common words to remove
nltk.download("punkt", quiet=True)        # Sentence/word tokenizer
nltk.download("wordnet", quiet=True)      # Word meanings database

from nltk.corpus import stopwords         # Import the stopwords list

# ---- Load spaCy English model ----
# spaCy loads a pre-trained English language model
# You must run: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")    # Small English model (~12MB)
    print("✅ spaCy model loaded successfully")
except OSError:
    print("❌ spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

# ---- Get English stopwords ----
# Stopwords = common words that don't carry meaning for our purpose
# Examples: "the", "is", "at", "which", "on"
STOP_WORDS = set(stopwords.words("english"))

# Add some extra resume-specific words that are also not useful
STOP_WORDS.update([
    "resume", "curriculum", "vitae", "cv", "dear", "sincerely",
    "objective", "summary", "experience", "education", "skill",
    "ability", "proficient", "responsible", "work", "year"
])


def clean_text(text: str) -> str:
    """
    Step 1-2: Basic text cleaning.

    Removes noise from raw text:
    - URLs (https://...)
    - Email addresses
    - Special characters and punctuation
    - Extra whitespace
    - Numbers (optional — comment out if you want to keep years like 2019)

    Args:
        text: Raw text from resume or job description

    Returns:
        Cleaned text string
    """
    if not text or not isinstance(text, str):
        return ""

    # Convert to lowercase so "Python" == "python"
    text = text.lower()

    # Remove URLs (http://... or https://...)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses (word@word.com)
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove special characters — keep only letters, spaces, and basic punctuation
    # \w = letters/numbers/underscore, \s = whitespace
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize_and_remove_stopwords(text: str) -> list:
    """
    Step 3-4: Split text into words and remove stopwords.

    TOKENIZATION = splitting "I love Python" → ["I", "love", "Python"]
    STOPWORD REMOVAL = remove words like "I", "the", "is"

    Args:
        text: Cleaned text string

    Returns:
        List of meaningful words
    """
    # Split the text into individual words
    words = text.split()

    # Keep a word only if:
    # - It's NOT a stopword
    # - It has more than 2 characters (skip very short words like "be", "do")
    filtered_words = [
        word for word in words
        if word not in STOP_WORDS and len(word) > 2
    ]

    return filtered_words


def lemmatize_with_spacy(words: list) -> list:
    """
    Step 5: Lemmatization using spaCy.

    LEMMATIZATION = reduce words to their base/dictionary form
    Examples:
        "running" → "run"
        "technologies" → "technology"
        "databases" → "database"
        "worked" → "work"

    This helps us match "developing" with "developer", etc.

    Args:
        words: List of words after stopword removal

    Returns:
        List of lemmatized words
    """
    if nlp is None:
        # If spaCy failed to load, return words as-is
        return words

    # Join words back into text for spaCy to process
    # (spaCy works better on full sentences than individual words)
    joined_text = " ".join(words)

    # Process with spaCy — this does tokenization, POS tagging, lemmatization
    doc = nlp(joined_text)

    # Extract the lemma (base form) of each token
    # We skip punctuation and spaces
    lemmas = [
        token.lemma_          # The base form of the word
        for token in doc
        if not token.is_punct  # Skip punctuation
        and not token.is_space  # Skip whitespace tokens
        and len(token.lemma_) > 2  # Skip very short lemmas
    ]

    return lemmas


def preprocess(text: str, lemmatize: bool = True) -> str:
    """
    MAIN FUNCTION: Full preprocessing pipeline.

    This is the function you'll call from other parts of the project.
    It runs all steps in order and returns clean, processed text.

    Pipeline:
        raw text → clean → tokenize → remove stopwords → lemmatize → join

    Args:
        text: Raw text (from resume or job description)
        lemmatize: Whether to apply lemmatization (default: True)

    Returns:
        Preprocessed text as a single string (space-separated tokens)

    Example:
        Input:  "Experienced Python Developer, working with Machine Learning!"
        Output: "experience python developer work machine learn"
    """
    # Step 1 & 2: Clean the text
    cleaned = clean_text(text)

    # Step 3 & 4: Tokenize and remove stopwords
    tokens = tokenize_and_remove_stopwords(cleaned)

    # Step 5: Lemmatize (optional but recommended)
    if lemmatize and nlp is not None:
        tokens = lemmatize_with_spacy(tokens)

    # Join tokens back into a single string for vectorization
    # TF-IDF needs text as a string, not a list
    result = " ".join(tokens)

    return result


def get_tokens(text: str) -> set:
    """
    Get unique tokens from preprocessed text.
    Used for skill gap analysis — to check which skills are present.

    Args:
        text: Raw text

    Returns:
        Set of unique processed tokens
    """
    processed = preprocess(text)
    return set(processed.split())


# ---- QUICK TEST ----
if __name__ == "__main__":
    sample = """
    Experienced Python Developer with 5 years of experience in Machine Learning
    and Deep Learning. Proficient in TensorFlow, PyTorch, and Scikit-learn.
    Strong background in SQL databases and data analysis.
    Email: john@example.com | LinkedIn: https://linkedin.com/in/john
    """

    print("=== Preprocessing Demo ===")
    print(f"\n📝 Original:\n{sample[:200]}")
    print(f"\n🧹 After clean_text:\n{clean_text(sample)[:200]}")
    print(f"\n🔤 After tokenize:\n{tokenize_and_remove_stopwords(clean_text(sample))[:20]}")
    print(f"\n✨ Final preprocessed:\n{preprocess(sample)[:300]}")
