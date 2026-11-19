"""Summarize development text coverage, language flags and tail lengths."""

from collections import Counter
import json
import unicodedata

import numpy as np
import pandas as pd

from phase2_sample import OUT, PUB, read_sample, write_rows
from writing_style import clean_body


def run():
    reviews = read_sample()
    flags = Counter()
    lengths = []
    for row in reviews:
        raw = row["text"]
        text = clean_body(raw)[0]
        letters = [c for c in text if c.isalpha()]
        other = sum("LATIN" not in unicodedata.name(c, "") for c in letters)
        flags["nonlatin_gt_10pct"] += bool(letters) and other/len(letters) > .1
        flags["contains_replacement_character"] += "\ufffd" in text
        flags["contains_url"] += "http://" in text.casefold() or "https://" in text.casefold()
        lengths.append(row["tokens"])
    distribution = {f"p{p}": float(np.percentile(lengths,p)) for p in (0,5,25,50,75,95,99,100)}
    distribution["reviews"] = len(lengths)
    distribution["over_1000_words"] = int(np.sum(np.array(lengths)>1000))
    (PUB/"text_quality.json").write_text(json.dumps({"flags":dict(flags),"raw_token_distribution":distribution},indent=2))
    context = pd.read_csv(OUT/"context.csv")
    write_rows(PUB/"context_distributions.csv",
               (dict(variable=c,nonmissing=int(context[c].count()),missing=int(context[c].isna().sum()),
                     p05=context[c].quantile(.05),p25=context[c].quantile(.25),median=context[c].median(),
                     p75=context[c].quantile(.75),p95=context[c].quantile(.95),maximum=context[c].max())
                for c in ("rating","review_age_days","helpful_votes","review_tokens","eligible_reviews","price")))
    print(flags,distribution)


if __name__ == "__main__":
    run()
