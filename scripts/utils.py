# ────────────────────────────────
# Imports
# ────────────────────────────────

import os
import re
import time
import pandas as pd
from IPython.display import display
import langid
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
try:
    from spacy_langdetect import LanguageDetector
    from spacy.language import Language
except ImportError:
    LanguageDetector = None 
from nltk.corpus import names


# ────────────────────────────────
# Global Config
# ────────────────────────────────

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


# ────────────────────────────────
# 1. File Handling
# ────────────────────────────────

def load_or_convert_to_csv(filename_no_ext):
    start = time.time()
    xlsx_path = os.path.join(RAW_DIR, f"{filename_no_ext}.xlsx")
    csv_path = os.path.join(RAW_DIR, f"{filename_no_ext}.csv")

    if os.path.exists(xlsx_path) and not os.path.exists(csv_path):
        print(f"🔄 Converting {filename_no_ext}.xlsx to CSV...")
        pd.read_excel(xlsx_path).to_csv(csv_path, index=False)
        print(f"✅ Saved converted CSV to: {csv_path}")
    else:
        print(f"📁 Loading from: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"✅ Data loaded: {filename_no_ext} ({len(df)} rows) (Time: {time.time() - start:.2f}s)")
    return df

def save_processed(df, filename):
    path = os.path.join(PROCESSED_DIR, filename)
    if filename.endswith(".csv"):
        df.to_csv(path, index=False)
    elif filename.endswith(".xlsx"):
        df.to_excel(path, index=False)
    print(f"💾 Saved to {path}")

def read_processed(filename):
    path = os.path.join(PROCESSED_DIR, filename)
    if filename.endswith(".csv"):
        df = pd.read_csv(path)
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(path)
    print(f"📁 Loaded from {path} ({len(df)} rows)")

# ────────────────────────────────
# 2. Logging time
# ────────────────────────────────

def log_time(start_time, message):
    elapsed = time.time() - start_time
    print(f"{message} (Time taken: {elapsed:.2f} seconds)")


# ────────────────────────────────
# 3. Duplicate Handling
# ────────────────────────────────

def remove_exact_duplicates(
    df, 
    subset, 
    sample_size=5, 
    verbose=True, 
    save_prefix=None
):
    """
    Removes exact duplicates based on subset. Returns cleaned DataFrame and removed duplicates.
    Saves cleaned result as CSV (for pipeline use) and audit file as Excel (for manual inspection).

    Parameters:
        df (pd.DataFrame): Input DataFrame
        subset (list[str]): Columns to use for duplicate detection
        sample_size (int): Number of sample rows to display
        verbose (bool): Print info and show preview
        save_prefix (str): Base name used for saving cleaned and audit files

    Returns:
        df_cleaned, df_removed
    """
    duplicate_mask = df.duplicated(subset=subset, keep='first')
    df_removed = df[duplicate_mask].copy()
    df_cleaned = df[~duplicate_mask].copy()

    if verbose and not df_removed.empty:
        print(f"🧹 Found and removed {len(df_removed)} duplicates based on {subset}")

        full_dupes = df[df.duplicated(subset=subset, keep=False)]
        kept = full_dupes[~duplicate_mask].assign(status="✅ Kept (original)")
        removed = full_dupes[duplicate_mask].assign(status="🗑 Removed (duplicate)")
        audit_df = pd.concat([kept, removed]).sort_values(subset + ["status"])

        print("🔍 Sample of duplicate rows (kept and removed):")
        display(audit_df[subset + ["status"]].head(sample_size * 2))

        if save_prefix:
            # Save cleaned CSV for pipeline
            cleaned_path = os.path.join(PROCESSED_DIR, f"{save_prefix}_cleaned.csv")
            df_cleaned.to_csv(cleaned_path, index=False)
            print(f"💾 Saved cleaned CSV to: {cleaned_path}")

            # Save audit Excel for manual inspection
            audit_path = os.path.join(PROCESSED_DIR, f"{save_prefix}_duplicates_audit.xlsx")
            audit_df.to_excel(audit_path, index=False)
            print(f"📝 Saved audit Excel to: {audit_path}")

    elif verbose:
        print(f"✅ No duplicates found based on {subset}")

    return df_cleaned, df_removed


# ────────────────────────────────
# 4. Text Cleaning
# ────────────────────────────────

def clean_text(doc):
    doc = doc.lower().replace("\n", " ")
    doc = re.sub(r"[/:=(){}\[\]\\|]", " ", doc)
    doc = re.sub(r"<.*?>", '', doc)
    doc = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in doc)

    tokens = word_tokenize(doc)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens).strip()


# ────────────────────────────────
# 5. Language Detection
# ────────────────────────────────


def detect_langdetect(text):
    try:
        return detect(text)
    except:
        return "unknown"

def is_english_langdetect(text):
    return detect_langdetect(text) == "en"

def is_english_langid(text):
    try:
        return langid.classify(text)[0] == "en"
    except:
        return False

if LanguageDetector:
    nlp_spacy_lang = spacy.load("en_core_web_sm")

    if "language_detector" not in nlp_spacy_lang.pipe_names:
        @Language.factory("language_detector")
        def create_lang_detector(nlp, name):
            return LanguageDetector()
        nlp_spacy_lang.add_pipe("language_detector", last=True)

    def detect_language_spacy(text):
        doc = nlp_spacy_lang(text)
        return doc._.language.get("language", "unknown")

else:
    def detect_language_spacy(text):
        print("⚠️ spacy_langdetect not installed.")
        return "unknown"
    

# ────────────────────────────────
# 6. Anonymization
# ────────────────────────────────

nlp_spacy_ner = spacy.load("en_core_web_sm")


def anonymize_text(text):
    if not isinstance(text, str):
        return text

    text = re.sub(r"\S+@\S+", "[EMAIL]", text)
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP]", text)
    text = re.sub(r"\b(?:\+?\d{1,3})?[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}\b", "[PHONE]", text)
    text = re.sub(r"\b\d{4,}\b", "[NUMBER]", text)

    doc = nlp_spacy_ner(text)
    new_text = text
    offset = 0

    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE"]:
            replacement = f"[{ent.label_}]"
            start = ent.start_char + offset
            end = ent.end_char + offset
            new_text = new_text[:start] + replacement + new_text[end:]
            offset += len(replacement) - (end - start)

    return new_text


def replace_name_patterns(text):
    if not isinstance(text, str):
        return text

    # Normalize whitespace first
    text = re.sub(r'\s+', ' ', text)

    # List of explicit names to replace (case-insensitive)
    name_list = ["Kevin", "Benjamin", "James", "Sallie", "Ryan"]
    name_pattern = r"(?i)\b(" + "|".join(name_list) + r")\b"
    text = re.sub(name_pattern, "[PERSON]", text)

    # Pattern-based replacements
    patterns = [
        r"(?i)(Caller:\s*)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
        r"(?i)(Spoke to\s*)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
        r"(?i)(Spoke with\s*)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",

        r"(?i)(Dear\s*)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)(?=[, ]?)",
        r"(?i)(Hi\s*)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)(?=[, ]?)",
        r"(?i)(Hello\s*)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)(?=[, ]?)",

        r"(?i)(Thanks,?\s+)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
        r"(?i)(Thanks,?\s*\n)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",

        r"(?i)(Best regards,?\s+)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
        r"(?i)(Best regards,?\s*\n)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",

        r"(?i)(Warm regards,?\s+)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
        r"(?i)(Warm regards,?\s*\n)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",

        r"(?i)(Kind regards,?\s+)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
        r"(?i)(Kind regards,?\s*\n)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",

        r"(?i)(Best,\s+)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
        r"(?i)(Best,\s*\n)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",

        r"(?i)(Respectfully,\s+)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
        r"(?i)(Respectfully,\s*\n)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",

        r"(?i)(My name is\s+)([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
    ]

    for pattern in patterns:
        text = re.sub(pattern, r"\1[PERSON]", text)

    return text

