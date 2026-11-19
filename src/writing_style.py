"""Measure observable style markers in English review bodies using fixed rules."""

import argparse
from collections import Counter
import gzip
from hashlib import blake2s
import html
import json
from math import sqrt
from pathlib import Path
import re

from audit import WORD


BLOCK_TAGS = {"br", "hr", "p", "div", "li", "ul", "ol", "blockquote"}
TAG = re.compile(r"</?(br|hr|p|div|li|ul|ol|blockquote|em|i|strong|b|u|a|span)\b[^>]*>", re.I)
END = re.compile(r"""(?P<marks>[.!?…]+)["'”’)\]]*(?=\s|$)|\n[^\S\n]*\n""")
ABBREVIATION = re.compile(r"\b(?i:mr|mrs|ms|dr|prof|sr|jr|st|vs|e\.g|i\.e)\.$|(?:\b[A-Z]\.)+$")
ELLIPSIS = re.compile(r"\.{3,}|…+")
PRONOUNS = {
    "first_singular": "i me my mine myself".split(),
    "first_plural": "we us our ours ourselves".split(),
    "second_person": "you your yours yourself yourselves".split(),
}
CONTRACTIONS = set("""
    i'm you're he's she's it's we're they're that's there's here's what's who's where's
    how's let's i've you've we've they've i'll you'll he'll she'll it'll we'll they'll
    i'd you'd he'd she'd it'd we'd they'd aren't isn't wasn't weren't haven't hasn't hadn't
    can't couldn't don't doesn't didn't won't wouldn't shan't shouldn't mustn't needn't
    mightn't ain't could've would've should've might've must've
""".split())


def tag_spacing(match):
    tag = match[1].lower()
    if tag == "br":
        return "\n"
    return "\n\n" if tag in BLOCK_TAGS else ""


def clean_body(body):
    decoded = html.unescape(body).replace("\r\n", "\n").replace("\r", "\n")
    quote_marker = bool(re.search(r'["“”]|<blockquote\b', decoded, re.I))
    cleaned = TAG.sub(tag_spacing, decoded)
    return cleaned.replace("’", "'").replace("‘", "'"), quote_marker


def sentence_parts(text):
    """Yield candidate sentence text and its terminal marks using the pilot boundaries."""
    start = 0
    for boundary in END.finditer(text):
        marks = boundary["marks"] or ""
        prefix = text[max(start, boundary.start() - 20):boundary.start() + 1]
        if marks == "." and ABBREVIATION.search(prefix):
            continue
        if ELLIPSIS.fullmatch(marks):
            continue
        yield text[start:boundary.start()], marks
        start = boundary.end()
    yield text[start:], ""


def basic_measurements(text, quote_marker, tokens, units):
    words, sentences = len(tokens), len(units)
    # Keep an apostrophe word as one token, but count the person in "I'm" or "you'll".
    heads = Counter(word.partition("'")[0] if word in CONTRACTIONS else word for word in tokens)
    counts = {name: sum(heads[word] for word in forms) for name, forms in PRONOUNS.items()}
    counts.update(contractions=sum(word in CONTRACTIONS for word in tokens),
                  exclamations=text.count("!"), ellipses=len(ELLIPSIS.findall(text)),
                  colons=text.count(":"), semicolons=text.count(";"))
    sum_sq = sum(length * length for length, _ in units)
    questions = sum("?" in marks for _, marks in units)
    rates = {name + "_per_1000_words": 1000 * count / words if words else None
             for name, count in counts.items()}
    return {
        "words": words, "sentences": sentences, "sentence_words_squared_sum": sum_sq,
        "question_endings": questions, "has_quote_marker": quote_marker, **counts, **rates,
        "mean_sentence_words": words / sentences if sentences else None,
        "sd_sentence_words": sqrt(sentences * sum_sq - words**2) / sentences if sentences >= 2 else None,
        "question_ending_pct": 100 * questions / sentences if sentences else None,
    }


def measure_style(body):
    """Return per-review counts, denominators and derived metrics; undefined rates are None."""
    text, quote_marker = clean_body(body)
    tokens = [word.casefold() for word in WORD.findall(text)]
    units = [(n, marks) for part, marks in sentence_parts(text) if (n := len(WORD.findall(part)))]
    return basic_measurements(text, quote_marker, tokens, units)


def pilot(snapshot):
    tally = Counter()
    users = set()
    with gzip.open(snapshot, "rb") as handle:
        for line in handle:
            record = json.loads(line)
            if record["split"] != "train":
                continue
            # A fixed 1/256 sample of development reviews, independent of body and genre.
            key = str(record["source_row"]).encode("ascii")
            if blake2s(key, digest_size=2).digest()[0] != 0:
                continue
            result = measure_style(record["text"])
            tally["reviews"] += 1
            users.add(record["user_key"])
            tally["words"] += result["words"]
            tally["sentences"] += result["sentences"]
            tally["reviews_with_questions"] += result["question_endings"] > 0
            tally["reviews_with_quote_marker"] += result["has_quote_marker"]
            tally["reviews_with_contractions"] += result["contractions"] > 0
            tally["reviews_with_undefined_sentence_sd"] += result["sd_sentence_words"] is None
            tally["reviews_with_no_words"] += result["words"] == 0
    return {**tally, "users": len(users)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--text", help="Measure one review supplied as a string.")
    mode.add_argument("--pilot", type=Path, help="Run the reproducible measurement pilot on the snapshot.")
    args = parser.parse_args()
    result = measure_style(args.text) if args.text is not None else pilot(args.pilot)
    print(json.dumps(result, indent=2))
