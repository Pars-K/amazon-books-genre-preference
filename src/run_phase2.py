"""Rebuild or verify the Phase 2 deliverables, with provenance and a command log."""

import argparse
from hashlib import file_digest, sha256
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from scipy import sparse
from phase2_sample import BASE, OUT, PUB, read_sample
from phase2_transform import transform

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path("reports/phase2")
MANIFEST = REPORT / "completion_manifest.json"
COMMANDS = (
    ["src/phase2_sample.py","sample"],
    ["src/phase2_sample.py","metadata"],
    ["src/phase2_sample.py","independent"],
    ["src/phase2_patterns.py"],
    ["src/phase2_reliability.py"],
    ["src/phase2_sensitivity.py"],
    ["src/phase2_interpret.py"],
    ["src/phase2_common_book.py"],
    ["src/phase2_quality.py"],
    ["src/phase2_figures.py"],
    ["src/build_phase2_report.py"],
    ["-m","unittest","discover","-s","tests"],
)


def hash_file(path, logical_source=False):
    if logical_source:
        return sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()
    with path.open("rb") as f:
        return file_digest(f,"sha256").hexdigest()


def checks():
    tests = subprocess.run([sys.executable,"-m","unittest","discover","-s","tests"],
                           capture_output=True,text=True,encoding="utf-8")
    output = tests.stdout+tests.stderr
    if tests.returncode:
        raise RuntimeError("Project checks failed:\n"+output)
    test_count = re.search(r"Ran (\d+) tests",output)
    if test_count is None:
        raise RuntimeError("The project test result is missing")
    original = subprocess.check_output(["git","show","phase-1:PROJECT_CARD.md"]).decode("utf-8")
    current = Path("PROJECT_CARD.md").read_text(encoding="utf-8")
    questions = lambda text: re.findall(r"^\d+\. \*\*.*$",text,re.M)
    if len(questions(original))!=10 or questions(original)!=questions(current):
        raise ValueError("Original ten Phase 1 questions differ")
    old_pdf = subprocess.check_output(["git","show","phase-1:reports/phase1.pdf"])
    if old_pdf != Path("reports/phase1.pdf").read_bytes():
        raise ValueError("Phase 1 PDF changed")
    for name in ("audit.py","genre_coverage.py","review_text_audit.py","review_quality_audit.py"):
        old = subprocess.check_output(["git","show","phase-1:src/"+name])
        now = Path("src",name).read_bytes()
        if old.replace(b"\r\n",b"\n") != now.replace(b"\r\n",b"\n"):
            raise ValueError("Phase 1 source changed: "+name)
    sample = read_sample()
    users = pd.read_csv(BASE/"users.csv",dtype={"user_key":str})
    roster = set(users.loc[users.split=="train","user_key"])
    assignment = pd.read_csv(OUT/"assignments.csv",dtype={"user_key":str})
    if [r["user_key"] for r in sample] != list(assignment.user_key):
        raise ValueError("Lexical rows do not match selected authors")
    if set(assignment.user_key)!=roster or assignment.user_key.duplicated().any():
        raise ValueError("Development roster mismatch")
    x = sparse.load_npz(OUT/"tfidf.npz")
    fitted = np.load(OUT/"lexical_fit.npz",allow_pickle=False)
    np.testing.assert_allclose(x.multiply(x).sum(1),1,atol=1e-10)
    np.testing.assert_allclose(fitted["components"]@fitted["components"].T,np.eye(48),atol=1e-10)
    first = pd.read_csv(OUT/"first_review_style.csv.gz",dtype={"user_key":str}).set_index("user_key")
    for i in np.random.default_rng(61).choice(len(sample),25,replace=False):
        result = transform(sample[i]["text"])
        np.testing.assert_allclose(result["svd_coordinates"],fitted["coordinates"][i],atol=1e-10)
        for key,value in result["style"].items():
            saved = first.loc[sample[i]["user_key"],key]
            if value is None:
                if not pd.isna(saved):
                    raise ValueError("Missing-value handling differs")
            elif not np.isclose(value,saved,atol=1e-6):
                raise ValueError("Review style extraction differs")
    evidence = pd.read_csv(OUT/"independent.csv",dtype={"user_key":str},keep_default_na=False)
    if evidence.user_key.duplicated().any() or set(evidence.user_key)!=roster:
        raise ValueError("Independent evidence roster mismatch")
    for path in (REPORT/"report-fa.md",REPORT/"README.md"):
        for target in re.findall(r"\]\(([^)]+)\)",path.read_text(encoding="utf-8")):
            if not target.startswith(("http:","https:","#")) and not (path.parent/target).resolve().exists():
                raise ValueError("Missing report link: "+target)
    return dict(unit_tests_passed=int(test_count[1]),development_users=len(roster),lexical_shape=list(x.shape),frozen_transform_checks=25,
                phase1_questions_preserved=10,phase1_pdf_preserved=True,phase1_audit_code_preserved=True)


def seal():
    audit = checks()
    inputs = [Path("data/raw/Books.jsonl.gz"),Path("data/raw/meta_Books.jsonl.gz"),
              BASE/"manifest.json",BASE/"users.csv",BASE/"memberships.csv",BASE/"reviews.jsonl.gz",
              Path("data/genre_coverage/book_genres.csv")]
    sources = [*sorted(Path("src").glob("*.py")),*sorted(Path("tests").glob("*.py")),Path("requirements.txt")]
    outputs = [*sorted(OUT.glob("*")),*sorted(PUB.glob("*")),REPORT/"report-fa.md",Path("reports/phase2.pdf")]
    state = dict(created_utc=datetime.now(timezone.utc).isoformat(),checks=audit,
                 runtime={name:importlib.metadata.version(name) for name in ("numpy","scipy","pandas","matplotlib")},
                 python=sys.version.split()[0],analysis_base_commit="3d3162a",
                 inputs={str(p):hash_file(p) for p in inputs},
                 source_sha256_lf={str(p):hash_file(p,True) for p in sources},
                 outputs={str(p):hash_file(p) for p in outputs if p.is_file()})
    MANIFEST.write_text(json.dumps(state,ensure_ascii=True,indent=2)+"\n",encoding="utf-8")
    print("Sealed Phase 2",audit,flush=True)


def verify():
    state = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for section,logical in (("inputs",False),("source_sha256_lf",True),("outputs",False)):
        for name,expected in state[section].items():
            if hash_file(Path(name),logical) != expected:
                raise ValueError("Changed file: "+name)
    checks()
    print("Verified Phase 2 hashes, authors, transforms and original Phase 1 materials",flush=True)


def rebuild():
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS="2",OMP_NUM_THREADS="2",PYTHONIOENCODING="utf-8")
    with (REPORT/"run.log").open("w",encoding="utf-8") as log:
        for command in COMMANDS:
            heading = datetime.now(timezone.utc).isoformat()+" "+sys.executable+" "+" ".join(command)
            print(heading,flush=True)
            log.write(heading+"\n")
            log.flush()
            with subprocess.Popen([sys.executable,*command],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  env=environment,text=True,encoding="utf-8") as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line,end="",flush=True)
                if process.wait():
                    raise RuntimeError("Stage failed: "+" ".join(command))
    seal()


if __name__ == "__main__":
    os.chdir(ROOT)
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rebuild",action="store_true")
    mode.add_argument("--seal",action="store_true")
    mode.add_argument("--verify",action="store_true")
    args = p.parse_args()
    if args.rebuild:
        rebuild()
    elif args.seal:
        seal()
    else:
        verify()
