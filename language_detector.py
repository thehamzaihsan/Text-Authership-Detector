"""
language_detector.py
--------------------
Detects whether a single word is English or Roman Urdu.

Returns a full detail dict:
    {
        "label"      : "english" | "roman_urdu" | "ambiguous" | "unknown",
        "confidence" : float (0.0 – 1.0),
        "matched"    : str | None,   # closest word found in a dictionary
        "method"     : str           # how the decision was made
    }

Install dependencies:
    pip install rapidfuzz nltk
    python -c "import nltk; nltk.download('words')"
"""

import re
import unicodedata
from functools import lru_cache
from rapidfuzz import process, fuzz
import nltk
from nltk.corpus import words as nltk_words

# ── ENGLISH VOCABULARY (large — from NLTK) ───────────────────────────────────
def _build_english_vocab() -> frozenset[str]:
    try:
        return frozenset(w.lower() for w in nltk_words.words())
    except LookupError:
        nltk.download("words", quiet=True)
        return frozenset(w.lower() for w in nltk_words.words())

ENGLISH_VOCAB: frozenset[str] = _build_english_vocab()
# List form needed for rapidfuzz (only built once)
ENGLISH_VOCAB_LIST: list[str] = list(ENGLISH_VOCAB)

# ── ROMAN URDU VOCABULARY (small, curated) ───────────────────────────────────
# Add / expand this list freely. Spelling here should be the most common form.
ROMAN_URDU_VOCAB: frozenset[str] = frozenset([
    # Pronouns & people
    "ap", "aps", "apko", "apka", "apki", "apne", "apna", "apni",
    "hm", "hum", "humko", "humara", "hamara", "hamari", "hamary",
    "wo", "woh", "unko", "unka", "unki", "unse", "unhone",
    "me", "mujhe", "mera", "meri", "mere", "main",
    "tu", "tujhe", "tera", "teri", "tere",
    "ye", "yeh", "isko", "iska", "iski", "isse", "inhone",
    "koi", "kisi", "kisiko", "sab", "sb",
    # Verbs (common forms)
    "he", "hai", "hain", "tha", "thi", "the",
    "ho", "hoga", "hogi", "honge",
    "kr", "kro", "krna", "karna", "krte", "krta", "krti", "kren", "karen",
    "krdia", "krdiya", "krdo", "kardiya",
    "ja", "jao", "jana", "jata", "jati", "jate",
    "aa", "aao", "aana", "aata", "aati", "aate", "aya", "ayi",
    "de", "do", "dena", "deta", "deti", "dete", "dedo", "dedena",
    "le", "lo", "lena", "leta", "leti", "lete",
    "bol", "bolo", "bolna", "bolta", "bolti",
    "sun", "suno", "sunna", "sunta", "sunti",
    "smjh", "samajh", "samjho", "samajhna",
    "bta", "btao", "batao", "batana", "btayen", "btaden",
    "rho", "raho", "rehna", "rehta", "rehti", "rehn",
    "rkho", "rakho", "rakhna",
    "dekho", "dekh", "dekhna", "dekhta", "dekhti",
    "milna", "milo", "milte", "milta",
    "socho", "soch", "sochna",
    "poch", "pocho", "puchna", "pochna",
    "lga", "lage", "lagta", "lagti", "lagana",
    "chal", "chalo", "chalna",
    "baith", "baithna", "baitho",
    "uth", "utho", "uthna",
    "khao", "khana", "khata", "khati",
    "pio", "pina", "pita", "piti",
    # Negation & affirmation
    "nhi", "nahi", "na", "mat", "han", "haan", "ji", "bilkul",
    "haan", "hna", "hn",
    # Question words
    "kia", "kya", "kyun", "kyunke", "kese", "kaise", "kab", "kahan",
    "kitna", "kitni", "kitne", "kaun", "kon",
    # Connectors & fillers
    "or", "aur", "lekin", "magar", "phir", "fer", "toh", "to",
    "bhi", "hi", "sirf", "bas", "bs", "agar", "jab", "jaise",
    "warna", "wese", "waise", "otherwise", "baqi", "baaki",
    "phir", "phr", "uske", "uski", "uska", "usme", "usne",
    "islie", "isliye", "kyunki", "chunke",
    "ke", "ki", "ka", "ko", "se", "me", "mein", "par", "pe",
    "sath", "saath", "stg", "sthy",
    # Common words & expressions
    "yaar", "yr", "bhai", "bro", "dost",
    "thik", "theek", "theek", "acha", "accha", "achha",
    "zyada", "ziyada", "kam", "kum",
    "abhi", "abi", "ab", "baad", "pehle", "phele",
    "kal", "aj", "aaj", "parso",
    "subah", "sham", "raat", "din",
    "ghar", "kaam", "kam", "paisa", "paise", "paisay",
    "time", "waqt", "jaldi", "der", "deri",
    "bht", "bohat", "bahut", "bara", "bada", "chota",
    "alag", "same", "sab", "sb",
    "project", "kaam", "kam",
    "btw", "waise", "vese",
    "inshallah", "mashallah", "alhamdulillah", "jazakallah",
    "chlen", "chalen", "chalein",
    "lelo", "lelena", "dedo", "dedena",
    "pta", "pata", "maloom", "معلوم",
    "tension", "fikar", "pareshan",
    "jitna", "utna", "itna",
    "zaroor", "zarur", "lazmi",
    "pehle", "baad", "ander", "bahar", "upar", "neeche",
    "shukriya", "shukria", "mehrbani",
    "khud", "apne aap",
    "matlab", "yani", "yaani",
    "seedha", "sidha", "sidha",
    "poora", "pura", "adha",
    "naya", "purana", "achi", "buri",
])
ROMAN_URDU_VOCAB_LIST: list[str] = list(ROMAN_URDU_VOCAB)

# ── AMBIGUOUS WORDS ───────────────────────────────────────────────────────────
# Words that exist meaningfully in both languages — never force a label
AMBIGUOUS: frozenset[str] = frozenset([
    "ok", "okay", "no", "hi", "na", "bus", "mat", "par", "se",
    "to", "so", "me", "he", "her", "us", "are", "but", "or",
    "din", "sir", "man", "yes", "hi", "ho", "go", "do", "be",
    "call", "time", "date", "deal", "list", "plan", "file",
    "good", "bad", "open", "use", "set", "main", "just",
])

# ── PHONETIC NORMALIZER (internal only, never exposed in output) ──────────────
def _phonetic_normalize(word: str) -> str:
    """
    Reduce a Roman Urdu word to a canonical phonetic form for lookup.
    This is ONLY used internally for matching — never stored or returned.
    """
    w = word.lower().strip()

    # Remove consecutive duplicate letters (krrna → krna, aachha → acha)
    w = re.sub(r'(.)\1+', r'\1', w)

    # Common vowel expansions / contractions
    w = w.replace("aa", "a").replace("ee", "i").replace("oo", "u")
    w = w.replace("ae", "e").replace("ai", "e")
    w = w.replace("ou", "u").replace("au", "a")

    # Common consonant equivalences
    w = w.replace("ph", "f")
    w = w.replace("gh", "g")
    w = w.replace("kh", "k")
    w = w.replace("ch", "c")
    w = w.replace("sh", "s")
    w = w.replace("wh", "w")
    w = w.replace("ck", "k")
    w = w.replace("qu", "k")
    w = w.replace("x",  "ks")

    # q → k, v → w, z → j (common Roman Urdu swaps)
    w = w.replace("q", "k")
    w = w.replace("v", "w")

    # Drop trailing silent characters
    w = w.rstrip("h") if len(w) > 3 else w

    # Drop short vowels between consonants (karna → krna style)
    # Only when word is all-consonant-heavy (> 60% consonants)
    vowels = set("aeiou")
    consonants = [c for c in w if c not in vowels and c.isalpha()]
    if len(w) > 0 and len(consonants) / len(w) > 0.5:
        # Remove interior short vowels cautiously
        w = re.sub(r'(?<=[bcdfghjklmnprstvwyz])[aeiou](?=[bcdfghjklmnprstvwyz])', '', w)

    return w

def _normalize_roman_urdu_vocab() -> dict[str, str]:
    """Pre-compute phonetic keys for the entire Roman Urdu vocab."""
    return {_phonetic_normalize(w): w for w in ROMAN_URDU_VOCAB_LIST}

PHONETIC_URDU_MAP: dict[str, str] = _normalize_roman_urdu_vocab()

# ── FUZZY THRESHOLDS ─────────────────────────────────────────────────────────
def _get_threshold(word_len: int) -> int:
    if word_len <= 3: return 100   # short words must be exact
    if word_len <= 5: return 88
    if word_len <= 8: return 82
    return 78

# ── CORE CLASSIFIER ──────────────────────────────────────────────────────────
@lru_cache(maxsize=4096)
def detect_language(word: str) -> dict:
    """
    Classify a single word as english / roman_urdu / ambiguous / unknown.

    Returns:
        {
            "label"      : str,
            "confidence" : float,
            "matched"    : str | None,
            "method"     : str
        }
    """
    # ── Sanitize input ────────────────────────────────────────────────────
    raw   = word.strip()
    clean = re.sub(r'[^\w]', '', raw).lower()   # strip punctuation, lowercase

    if not clean or not clean.isalpha():
        return {"label": "unknown", "confidence": 0.0,
                "matched": None, "method": "non_alpha_input"}

    # ── 1. Ambiguous list (check first, highest priority) ─────────────────
    if clean in AMBIGUOUS:
        return {"label": "ambiguous", "confidence": 1.0,
                "matched": clean, "method": "ambiguous_list"}

    # ── 2. Exact English match ────────────────────────────────────────────
    if clean in ENGLISH_VOCAB:
        return {"label": "english", "confidence": 1.0,
                "matched": clean, "method": "exact_english"}

    # ── 3. Exact Roman Urdu match ─────────────────────────────────────────
    if clean in ROMAN_URDU_VOCAB:
        return {"label": "roman_urdu", "confidence": 1.0,
                "matched": clean, "method": "exact_roman_urdu"}

    # ── 4. Phonetic Roman Urdu match (internal only) ──────────────────────
    phonetic_key = _phonetic_normalize(clean)
    if phonetic_key in PHONETIC_URDU_MAP:
        matched_original = PHONETIC_URDU_MAP[phonetic_key]
        return {"label": "roman_urdu", "confidence": 0.92,
                "matched": matched_original, "method": "phonetic_roman_urdu"}

    # ── 5. Fuzzy English match ────────────────────────────────────────────
    threshold = _get_threshold(len(clean))
    eng_result = process.extractOne(
        clean, ENGLISH_VOCAB_LIST,
        scorer=fuzz.ratio,
        score_cutoff=threshold
    )
    if eng_result:
        matched_word, score, _ = eng_result
        confidence = round(score / 100, 2)
        return {"label": "english_misspelled", "confidence": confidence,
                "matched": matched_word, "method": "fuzzy_english"}

    # ── 6. Fuzzy Roman Urdu match ─────────────────────────────────────────
    urdu_result = process.extractOne(
        clean, ROMAN_URDU_VOCAB_LIST,
        scorer=fuzz.ratio,
        score_cutoff=threshold
    )
    if urdu_result:
        matched_word, score, _ = urdu_result
        confidence = round(score / 100, 2)
        return {"label": "roman_urdu_variant", "confidence": confidence,
                "matched": matched_word, "method": "fuzzy_roman_urdu"}

    # ── 7. Phonetic fuzzy Roman Urdu (last resort) ────────────────────────
    phonetic_keys = list(PHONETIC_URDU_MAP.keys())
    ph_result = process.extractOne(
        phonetic_key, phonetic_keys,
        scorer=fuzz.ratio,
        score_cutoff=75
    )
    if ph_result:
        matched_key, score, _ = ph_result
        matched_original = PHONETIC_URDU_MAP[matched_key]
        confidence = round((score / 100) * 0.85, 2)  # penalize indirect match
        return {"label": "roman_urdu_variant", "confidence": confidence,
                "matched": matched_original, "method": "phonetic_fuzzy_roman_urdu"}

    # ── 8. Unknown ────────────────────────────────────────────────────────
    return {"label": "unknown", "confidence": 0.0,
            "matched": None, "method": "no_match"}


# ── LABEL GROUPS (helpers for downstream use) ─────────────────────────────────
ENGLISH_LABELS    = {"english", "english_misspelled"}
ROMAN_URDU_LABELS = {"roman_urdu", "roman_urdu_variant"}

def is_english(result: dict)    -> bool: return result["label"] in ENGLISH_LABELS
def is_roman_urdu(result: dict) -> bool: return result["label"] in ROMAN_URDU_LABELS
def is_ambiguous(result: dict)  -> bool: return result["label"] == "ambiguous"
def is_unknown(result: dict)    -> bool: return result["label"] == "unknown"


# ── QUICK TEST ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_words = [
        "hello", "krna", "karna", "okay", "btao", "project",
        "inshallah", "running", "runnin", "samjho", "smjh",
        "acha", "achha", "bus", "developer", "develper", "nahi", "nhi",
        "yaar", "yr", "hain", "are", "documentation", "documntation",
    ]
    print(f"\n{'Word':<18} {'Label':<22} {'Conf':>5}  {'Matched':<20} Method")
    print("-" * 80)
    for w in test_words:
        r = detect_language(w)
        print(f"{w:<18} {r['label']:<22} {r['confidence']:>5.2f}  "
              f"{str(r['matched']):<20} {r['method']}")