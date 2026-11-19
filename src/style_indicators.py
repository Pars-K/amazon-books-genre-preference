"""Fixed English markers, additive statistics, and their review/user denominators."""

from collections import Counter, defaultdict
from math import sqrt
import re

from audit import WORD
from writing_style import basic_measurements, clean_body, sentence_parts


WINDOW = 20
EMPHATIC_RUN = re.compile(r"[!?]{2,}")
PARAGRAPH = re.compile(r"\n[^\S\n]*\n")
# Each category is counted at most once per sentence, even with multiple matches.
MARKERS = {
    "subordination": ("because", "although", "though", "unless", "whenever", "whereas",
                      "while", "since", "if", "when", "even though", "as soon as"),
    "hedge": ("perhaps", "maybe", "possibly", "probably", "i think", "i suspect",
              "it seems", "it seemed", "in my opinion", "might be", "may be"),
    "certainty": ("definitely", "certainly", "undoubtedly", "without doubt", "without a doubt",
                  "no doubt", "there is no question"),
    "causal": ("because", "therefore", "thus", "as a result", "for this reason", "that is why"),
    "contrast": ("but", "however", "nevertheless", "nonetheless", "on the other hand",
                 "in contrast", "whereas", "although"),
    "example": ("for example", "for instance", "such as", "to illustrate"),
    "reading_experience": ("i read", "i've read", "i reread", "when i read", "i finished",
                           "i started reading", "i was reading", "i couldn't put it down",
                           "i could not put it down", "kept me reading"),
    "recommendation": ("i recommend", "i highly recommend", "would recommend", "highly recommend",
                       "strongly recommend", "i don't recommend", "i do not recommend",
                       "i wouldn't recommend", "worth reading", "a must read"),
    "plot_description": ("the story follows", "the book follows", "the novel follows",
                         "tells the story of", "the story of", "is set in", "takes place in",
                         "the plot revolves around", "the story begins", "the story opens"),
    "evaluation": ("well written", "poorly written", "beautifully written", "a great read",
                   "an excellent read", "a good read", "i loved", "i enjoyed", "i hated",
                   "i disliked", "i liked", "well developed", "poorly developed"),
}
STARTS = defaultdict(list)
for category, phrases in MARKERS.items():
    for phrase in phrases:
        parts = phrase.split()
        STARTS[parts[0]].append((parts, category))

COLLOQUIAL = frozenset("gonna wanna gotta kinda sorta yeah yep nope lol omg btw".split())
ANCHORS = frozenset("the this that these those and with was were are have has of it my "
                    "you your is but for from they their which".split())
BASIC = ("words", "sentences", "sentence_words_squared_sum", "question_endings", "first_singular",
         "first_plural", "second_person", "contractions", "exclamations", "ellipses", "colons", "semicolons")
EXTRA = ("reviews", "paragraphs", "quote_reviews", "no_english_anchor_reviews", "windows20",
         "distinct_in_windows", "upper_tokens", "emphatic_runs", "colloquial_tokens")
COUNTS = BASIC + EXTRA + tuple(name + "_sentences" for name in MARKERS)

# (numerator, denominator, scale); one review and one user use exactly the same definitions.
RATIOS = {
    "words_per_review": ("words", "reviews", 1),
    "sentences_per_review": ("sentences", "reviews", 1),
    "paragraphs_per_review": ("paragraphs", "reviews", 1),
    "words_per_paragraph": ("words", "paragraphs", 1),
    "mean_sentence_words": ("words", "sentences", 1),
    "question_ending_pct": ("question_endings", "sentences", 100),
    "vocabulary_diversity_20_pct": ("distinct_in_windows", "windows20", 100 / WINDOW),
}
for name in ("first_singular", "first_plural", "second_person", "contractions", "exclamations",
             "ellipses", "colons", "semicolons", "upper_tokens", "emphatic_runs", "colloquial_tokens"):
    RATIOS[name + "_per_1000_words"] = (name, "words", 1000)
for name in MARKERS:
    RATIOS[name + "_sentence_pct"] = (name + "_sentences", "sentences", 100)
METRICS = tuple(RATIOS) + ("sd_sentence_words",)


def sentence_markers(tokens):
    found = set()
    for i, word in enumerate(tokens):
        for parts, category in STARTS.get(word, ()):
            if len(parts) == 1 or tokens[i:i + len(parts)] == parts:
                found.add(category)
    return found


def counts_from_text(body):
    text, quoted = clean_body(body)
    tokens, units = [], []
    markers = Counter()
    uppercase = 0
    for part, marks in sentence_parts(text):
        originals = WORD.findall(part)
        if not originals:
            continue
        units.append((len(originals), marks))
        uppercase += sum(len(t) >= 2 and t.isalpha() and t.isupper() for t in originals)
        words = [t.casefold() for t in originals]
        markers.update(sentence_markers(words))
        tokens.extend(words)
    basic = basic_measurements(text, quoted, tokens, units)
    result = {name: basic[name] for name in BASIC}
    windows = len(tokens) // WINDOW
    result.update(
        reviews=1, paragraphs=sum(bool(WORD.search(p)) for p in PARAGRAPH.split(text)),
        quote_reviews=int(quoted), no_english_anchor_reviews=int(ANCHORS.isdisjoint(tokens)),
        windows20=windows,
        distinct_in_windows=sum(len(set(tokens[i:i + WINDOW])) for i in range(0, windows * WINDOW, WINDOW)),
        upper_tokens=uppercase, emphatic_runs=len(EMPHATIC_RUN.findall(text)),
        colloquial_tokens=sum(t in COLLOQUIAL for t in tokens),
    )
    result.update({name + "_sentences": markers[name] for name in MARKERS})
    return result


def metrics(counts):
    values = {key: counts[n] * scale / counts[d] if counts[d] else None
              for key, (n, d, scale) in RATIOS.items()}
    n = counts["sentences"]
    values["sd_sentence_words"] = (
        sqrt(n * counts["sentence_words_squared_sum"] - counts["words"] ** 2) / n if n >= 2 else None)
    return values
