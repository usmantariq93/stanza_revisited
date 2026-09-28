"""Re-evaluating two limitations of Stanza (Qi et al., 2020) with the current release.

RQ1 (genre robustness): single-treebank models (ewt, gum) vs. pooled model (combined),
    evaluated on four UD English test sets (gold tokens; UPOS / UAS / LAS).
RQ2 (speed): runtime of Stanza (CPU / GPU) relative to spaCy (CPU), as in Table 4 of the paper.

Usage:  pip install stanza spacy torch
        python -m spacy download en_core_web_sm
        python run_experiments.py            # writes results.json
"""
import json, os, time, urllib.request, platform, argparse

BASE = "https://raw.githubusercontent.com/UniversalDependencies/"
TESTSETS = {  # name -> path of the UD test file (master branch)
    "EWT":    "UD_English-EWT/master/en_ewt-ud-test.conllu",
    "GUM":    "UD_English-GUM/master/en_gum-ud-test.conllu",
    "LinES":  "UD_English-LinES/master/en_lines-ud-test.conllu",
    "ParTUT": "UD_English-ParTUT/master/en_partut-ud-test.conllu",
}
PACKAGES = ["ewt", "gum", "combined"]  # single, single, pooled


def fetch(name):
    os.makedirs("data", exist_ok=True)
    path = f"data/{name}.conllu"
    if not os.path.exists(path):
        urllib.request.urlretrieve(BASE + TESTSETS[name], path)
    return path


def read_conllu(path):
    """Returns list of sentences; each a list of (form, upos, head, deprel). Skips MWT ranges/empty nodes."""
    sents, cur = [], []
    for line in open(path, encoding="utf8"):
        line = line.rstrip("\n")
        if not line:
            if cur: sents.append(cur); cur = []
        elif not line.startswith("#"):
            c = line.split("\t")
            if "-" in c[0] or "." in c[0]:
                continue
            cur.append((c[1], c[3], int(c[6]), c[7].split(":")[0]))  # strip deprel subtype (as in CoNLL-18 script)
    if cur: sents.append(cur)
    return sents


def evaluate(package, sents, gpu):
    import stanza
    nlp = stanza.Pipeline("en", package=package, processors="tokenize,pos,lemma,depparse",
                          tokenize_pretokenized=True, use_gpu=gpu, verbose=False)
    doc = nlp([[w[0] for w in s] for s in sents])
    n = up = ua = la = 0
    for gold, pred in zip(sents, doc.sentences):
        assert len(gold) == len(pred.words), "token misalignment"
        for g, p in zip(gold, pred.words):
            n += 1
            up += g[1] == p.upos
            hit = g[2] == p.head
            ua += hit
            la += hit and g[3] == (p.deprel or "").split(":")[0]
    return {"upos": 100 * up / n, "uas": 100 * ua / n, "las": 100 * la / n, "tokens": n}


def time_stanza(sents, gpu, package="combined", reps=2):
    import stanza
    texts = [" ".join(w[0] for w in s) for s in sents]
    nlp = stanza.Pipeline("en", package=package, processors="tokenize,pos,lemma,depparse", use_gpu=gpu, verbose=False)
    nlp("\n\n".join(texts[:20]))  # warm-up
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter(); nlp("\n\n".join(texts)); best = min(best, time.perf_counter() - t)
    return best


def time_spacy(sents, reps=2):
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    texts = [" ".join(w[0] for w in s) for s in sents]
    list(nlp.pipe(texts[:20]))
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter(); list(nlp.pipe(texts, batch_size=64)); best = min(best, time.perf_counter() - t)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed_sents", type=int, default=1000)
    a = ap.parse_args()
    import torch, stanza
    has_gpu = torch.cuda.is_available()
    res = {"accuracy": {}, "speed": {}, "meta": {
        "stanza": stanza.__version__, "torch": torch.__version__, "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "gpu": torch.cuda.get_device_name(0) if has_gpu else None}}
    data = {n: read_conllu(fetch(n)) for n in TESTSETS}
    for pkg in PACKAGES:
        stanza.download("en", package=pkg, verbose=False)
        res["accuracy"][pkg] = {}
        for tb, sents in data.items():
            res["accuracy"][pkg][tb] = evaluate(pkg, sents, has_gpu)
            print(pkg, tb, res["accuracy"][pkg][tb], flush=True)
    sub = data["EWT"][: a.speed_sents]
    ntok = sum(len(s) for s in sub)
    times = {"spaCy sm (CPU)": time_spacy(sub), "Stanza (CPU)": time_stanza(sub, False)}
    if has_gpu:
        times["Stanza (GPU)"] = time_stanza(sub, True)
    for k, t in times.items():
        res["speed"][k] = {"seconds": t, "tokens_per_s": ntok / t, "relative_to_spacy": t / times["spaCy sm (CPU)"]}
    res["meta"]["speed_tokens"] = ntok
    json.dump(res, open("results.json", "w"), indent=2)
    print("saved results.json")
