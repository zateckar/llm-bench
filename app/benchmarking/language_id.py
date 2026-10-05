"""Deterministic language identification for answers (``language-id-v1``).

A small, offline detector for the languages people use with our models:
English, Czech, Slovak, German and Polish (Polish is detected so that
Czech-to-Polish drift is visible). It scores function words and
language-specific letters. It is a measurement instrument for answers the
suite asked for in a known language, not a general language identifier.
"""

from __future__ import annotations

import re
import unicodedata

REVISION = "language-id-v1"
LANGUAGES = ("en", "cs", "sk", "de", "pl")
NAMES = {"en": "English", "cs": "Czech", "sk": "Slovak", "de": "German", "pl": "Polish"}
MIN_SENTENCE_WORDS = 4
MIN_HITS = 3
MARGIN = 1.5

_WORDS = {
    # Words shared with another supported language (an, was, also, i, do, to) are left out.
    "en": """the and of is are were in that it for with as on this be by you your not or have has
             from at which will can would should there their they we our if but what when how these those
             been than into about please does""",
    "cs": """se je jsou jsem jste jsme být bude budou byl byla bylo není nejsou nebo také který která které
             kterou kteří při už jen když jejich jeho její váš vaše mohu můžete můžeme prosím děkuji
             protože pouze ještě však tento tato toto tyto což jak aby ale pro na ve že od do po
             za tak mezi než pokud během této tohoto proto velmi další všechny tedy jsme nám jiné""",
    "sk": """sa je sú som ste sme byť bude budú bol bola bolo nie nie sú alebo tiež ktorý ktorá ktoré
             ktorú ktorí pri už len keď ich jeho jej váš vaša môžem môžete môžeme prosím ďakujem
             pretože iba ešte však tento táto toto tieto čo ako aby ale pre na vo že od do po
             za tak medzi než aj ak počas tejto tohto preto veľmi ďalšie všetky teda lebo nám iné""",
    "de": """der die das und ist nicht ein eine einen einem zu den mit sich des auf für im dem sie es von
             auch wir ich werden wird bitte können sind oder aber bei nach wie noch ihr ihre haben kann
             dass wenn nur zum zur über unter diese dieser dieses sowie""",
    "pl": """w nie się na jest że z jak ale czy są dla tak jestem być który która które oraz
             przez może tylko już bardzo proszę dziękuję jego jej ich ten ta te jeśli albo więc gdy
             również jednak""",
}
LEXICON = {lang: frozenset(words.split()) for lang, words in _WORDS.items()}
# Letters that only (or almost only) one of the languages uses.
LETTERS = {
    "cs": frozenset("řůě"),
    "sk": frozenset("ľĺŕô"),
    "de": frozenset("ßöü"),
    "pl": frozenset("łąęśźżńć"),
}
SHARED_LETTERS = {"ä": ("sk", "de")}

_CODE_BLOCK = re.compile(r"```.*?(?:```|$)", re.S)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")
_SENTENCE = re.compile(r"(?<=[.!?…])\s+|\n+")


def clean(text):
    text = unicodedata.normalize("NFC", text or "")
    for pattern in (_CODE_BLOCK, _INLINE_CODE, _URL, _EMAIL):
        text = pattern.sub(" ", text)
    return text


def words(text):
    return [w.casefold() for w in _WORD.findall(text)]


def scores(tokens):
    out = dict.fromkeys(LANGUAGES, 0.0)
    for word in tokens:
        for lang in LANGUAGES:
            if word in LEXICON[lang]:
                out[lang] += 1
        letters = set(word)
        for lang, marks in LETTERS.items():
            if letters & marks:
                out[lang] += 1
        for letter, langs in SHARED_LETTERS.items():
            if letter in letters:
                for lang in langs:
                    out[lang] += 0.5
        # Slovak infinitives end in -ť (robiť, fungovať); Czech ones in -t.
        if len(word) > 3 and word.endswith("ť"):
            out["sk"] += 1
    return out


def classify_words(tokens):
    """``(language or None, scores)``: None when the evidence is thin or ambiguous."""
    result = scores(tokens)
    ranked = sorted(result.items(), key=lambda kv: -kv[1])
    (best, top), (_, second) = ranked[0], ranked[1]
    if top < MIN_HITS and not (len(tokens) < 8 and top >= 2 and second == 0):
        return None, result
    if second and top < MARGIN * second:
        return None, result
    return best, result


def detect(text):
    """Overall language, per-sentence breakdown and the share of words per language."""
    body = clean(text)
    tokens = words(body)
    language, overall = classify_words(tokens)
    sentences, counted = [], dict.fromkeys(LANGUAGES, 0)
    for raw in _SENTENCE.split(body):
        sentence_words = words(raw)
        if len(sentence_words) < MIN_SENTENCE_WORDS:
            continue
        lang, _ = classify_words(sentence_words)
        sentences.append({"text": raw.strip()[:160], "language": lang, "words": len(sentence_words)})
        if lang:
            counted[lang] += len(sentence_words)
    classified = sum(counted.values())
    return {
        "revision": REVISION,
        "language": language,
        "words": len(tokens),
        "scores": overall,
        "sentences": sentences,
        "shares": {lang: n / classified for lang, n in counted.items() if n} if classified else {},
    }


def adherence(text, required, min_share=0.85):
    """``(ok, detail)``: the answer is in ``required`` and does not drift."""
    found = detect(text)
    share = found["shares"].get(required, 1.0 if not found["shares"] else 0.0)
    ok = found["language"] == required and share >= min_share
    return ok, found, share
