"""Transform a new review with frozen development preprocessing."""

import argparse
import json
from pathlib import Path

import numpy as np

from phase2_patterns import terms, unit, vectorize
from phase2_sample import OUT
from style_indicators import counts_from_text, metrics


def transform(text, fitted_path=OUT / "lexical_fit.npz"):
    with np.load(fitted_path, allow_pickle=False) as fit:
        counts = terms(text)
        x = vectorize([counts], list(fit["vocabulary"]), fit["idf"])
        raw = np.asarray(x @ fit["components"].T)
        coordinates = unit(raw)
    style_counts = counts_from_text(text)
    return {
        "style": metrics(style_counts),
        "observed_words": style_counts["words"],
        "quote_marker": bool(style_counts["quote_reviews"]),
        "english_anchor_found": not bool(style_counts["no_english_anchor_reviews"]),
        "known_lexical_features": int(x.nnz),
        "lexical_features_found": len(counts),
        "svd_coordinates": raw[0].tolist(),
        "normalized_svd_coordinates": coordinates[0].tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text-file", type=Path)
    inputs.add_argument("--text")
    parser.add_argument("--fit", type=Path, default=OUT / "lexical_fit.npz")
    options = parser.parse_args()
    text = options.text_file.read_text(encoding="utf-8") if options.text_file else options.text
    result = transform(text, options.fit)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False, indent=2))
