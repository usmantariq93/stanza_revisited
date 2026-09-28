"""Builds the DIN A1 poster (841 x 594 mm, landscape) from results.json.
   python make_poster.py --authors "Your Name" --repo https://github.com/you/repo
Without results.json the results panel shows clearly marked placeholders (never invented numbers)."""
import json, os, argparse
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import Paragraph, Frame, Spacer, Image, Table, TableStyle, Flowable
from reportlab.pdfgen import canvas
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W, H = 841 * mm, 594 * mm
NAVY, RED, GREY, LIGHT = colors.HexColor("#14213D"), colors.HexColor("#C1121F"), colors.HexColor("#4A4A4A"), colors.HexColor("#F1F3F8")
PKG_COL = {"ewt": "#8D99AE", "gum": "#F4A261", "combined": "#14213D"}
BODY = ParagraphStyle("b", fontName="Helvetica", fontSize=24, leading=31, textColor=colors.black, alignment=TA_LEFT, spaceAfter=7)
BUL = ParagraphStyle("u", parent=BODY, leftIndent=18, bulletIndent=2, spaceAfter=5)
H2 = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=40, leading=46, textColor=NAVY, spaceAfter=8)
CAP = ParagraphStyle("c", parent=BODY, fontSize=24, leading=28, textColor=GREY)
TB = ParagraphStyle("t", parent=BODY, fontSize=24, leading=28, spaceAfter=0)

TREEBANKS = ["EWT", "GUM", "LinES", "ParTUT"]
OOD = ["LinES", "ParTUT"]  # not part of any Stanza 'combined' training set (verify in Stanza docs)


class Placeholder(Flowable):
    def __init__(self, w, h, text): super().__init__(); self.w, self.h, self.text = w, h, text
    def wrap(self, *a): return self.w, self.h
    def draw(self):
        c = self.canv; c.setStrokeColor(RED); c.setDash(8, 6); c.setLineWidth(3); c.rect(0, 0, self.w, self.h)
        c.setFillColor(RED); c.setFont("Helvetica-Bold", 26)
        c.drawCentredString(self.w / 2, self.h / 2, self.text)


def P(t, s=BODY): return Paragraph(t, s)
def B(t): return Paragraph(t, BUL, bulletText="\u2022")


def fig_accuracy(res, w_mm, h_mm, path):
    fig, ax = plt.subplots(figsize=(w_mm / 25.4, h_mm / 25.4))
    plt.rcParams.update({"font.size": 24})
    for i, pkg in enumerate(res["accuracy"]):
        ys = [res["accuracy"][pkg][t]["las"] for t in TREEBANKS]
        ax.plot(range(4), ys, "o-", ms=17, lw=4, color=PKG_COL.get(pkg, "k"), label={"combined": "combined (pooled)"}.get(pkg, pkg))
    ax.set_xticks(range(4)); ax.set_xticklabels(TREEBANKS, fontsize=24)
    ax.set_ylabel("LAS (gold tokens)", fontsize=24); ax.tick_params(axis="y", labelsize=24)
    ax.grid(axis="y", alpha=.3); ax.legend(fontsize=22, frameon=False, loc="lower left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_speed(res, w_mm, h_mm, path):
    sp = res["speed"]
    labels = ["Stanza v1.0\nCPU (2020)", "Stanza v1.0\nGPU (2020)"] + [k.replace(" (", "\n(") for k in sp if "spaCy" not in k]
    vals = [10.3, 3.22] + [v["relative_to_spacy"] for k, v in sp.items() if "spaCy" not in k]
    cols = ["#BBBBBB", "#BBBBBB"] + ["#14213D"] * (len(vals) - 2)
    fig, ax = plt.subplots(figsize=(w_mm / 25.4, h_mm / 25.4))
    ax.bar(range(len(vals)), vals, color=cols)
    for i, v in enumerate(vals): ax.text(i, v, f"{v:.1f}\u00d7", ha="center", va="bottom", fontsize=24)
    ax.axhline(1, color="#C1121F", lw=3); ax.text(len(vals) - .5, 1.05, "spaCy = 1", color="#C1121F", ha="right", va="bottom", fontsize=22)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=22)
    ax.set_ylabel("Runtime (spaCy CPU = 1)", fontsize=24); ax.tick_params(axis="y", labelsize=24)
    ax.set_ylim(0, max(vals) * 1.18)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def findings(res):
    if not res:
        return [P("<b>Findings are generated automatically from results.json</b> \u2013 run <i>run_experiments.py</i>, then re-run <i>make_poster.py</i>.")], None
    acc = res["accuracy"]
    ood = {p: sum(acc[p][t]["las"] for t in OOD) / len(OOD) for p in acc}
    allm = {p: sum(acc[p][t]["las"] for t in TREEBANKS) / len(TREEBANKS) for p in acc}
    best = max(ood, key=ood.get)
    h1 = best == "combined"
    out = [B(f"<b>RQ1:</b> On held-out treebanks (LinES, ParTUT) mean LAS is "
             + ", ".join(f"{ood[p]:.1f} ({p})" for p in acc) + f". H1 is <b>{'supported' if h1 else 'not supported'}</b>: best = <b>{best}</b>."),
           B("Macro-LAS over all four sets: " + ", ".join(f"{allm[p]:.1f} ({p})" for p in acc) + "."),]
    sp = res["speed"]
    st = ", ".join(f"{k}: {v['relative_to_spacy']:.1f}\u00d7" for k, v in sp.items() if "spaCy" not in k)
    out.append(B(f"<b>RQ2:</b> runtime relative to spaCy CPU \u2013 {st} (2020 paper: 10.3\u00d7 CPU, 3.22\u00d7 GPU)."))
    return out, h1


def results_table(res):
    hdr = ["Model"] + TREEBANKS + ["Macro"]
    rows = [hdr]
    for p in ["ewt", "gum", "combined"]:
        if res and p in res["accuracy"]:
            a = res["accuracy"][p]
            cells = [f"{a[t]['upos']:.1f} / {a[t]['las']:.1f}" for t in TREEBANKS]
            m = f"{sum(a[t]['upos'] for t in TREEBANKS)/4:.1f} / {sum(a[t]['las'] for t in TREEBANKS)/4:.1f}"
        else:
            cells, m = ["\u2013"] * 4, "\u2013"
        rows.append([p] + cells + [m])
    data = [[Paragraph(str(c), ParagraphStyle("x", parent=TB, fontName="Helvetica-Bold" if r == 0 else "Helvetica")) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[52 * mm] + [45 * mm] * 4 + [50 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), LIGHT), ("LINEBELOW", (0, 0), (-1, 0), 2, NAVY),
                           ("LINEBELOW", (0, -1), (-1, -1), 2, NAVY), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def frame_fill(c, x, y, w, h, flow, name):
    c.setFillColor(LIGHT); c.roundRect(x, y, w, h, 8 * mm, stroke=0, fill=1)
    f = Frame(x, y, w, h, leftPadding=10 * mm, rightPadding=10 * mm, topPadding=8 * mm, bottomPadding=6 * mm, showBoundary=0)
    left = list(flow); f.addFromList(left, c)
    if left: raise SystemExit(f"OVERFLOW in column '{name}' \u2013 shorten text ({len(left)} flowables left)")


def build(a):
    res = json.load(open(a.results)) if os.path.exists(a.results) else None
    c = canvas.Canvas(a.out, pagesize=(W, H))
    c.setTitle("Stanza revisited \u2013 research poster"); c.setAuthor(a.authors)
    # title band
    c.setFillColor(NAVY); c.rect(0, H - 105 * mm, W, 105 * mm, stroke=0, fill=1)
    c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 78)
    c.drawString(20 * mm, H - 52 * mm, "Stanza Revisited: Are the 2020 Limitations Resolved?")
    c.setFont("Helvetica", 36)
    c.drawString(20 * mm, H - 72 * mm, "Genre robustness and speed of a multilingual neural NLP toolkit, re-evaluated on Universal Dependencies")
    c.setFont("Helvetica", 30)
    c.drawString(20 * mm, H - 92 * mm, f"{a.authors}  |  NLP module, University of Trier  |  Code: {a.repo}")
    cw = [235 * mm, 235 * mm, 317 * mm]; gap = 12 * mm; m = 15 * mm
    top = H - 105 * mm - 12 * mm; hh = top - 15 * mm
    xs = [m, m + cw[0] + gap, m + cw[0] + cw[1] + 2 * gap]

    hyp = [P("1  Hypothesis", H2),
           P("<b>Context.</b> Stanza (Qi et al., 2020) offers a fully neural pipeline (tokenization to dependency parsing) for 66 languages, trained on Universal Dependencies (UD) treebanks."),
           P("<b>Two limitations named by the authors:</b>"),
           B("<b>L1 \u2013 Single-dataset models:</b> most released models are trained on one dataset and may not transfer across genres."),
           B("<b>L2 \u2013 Speed:</b> accuracy-first design; on CPU the UD pipeline needed 10.3\u00d7 spaCy's runtime (their Table 4)."),
           P("<b>Since then:</b> Stanza's default English models were retrained on pooled treebanks (EWT + GUM, release v1.2), and English UD data have been revised and harmonised (Zeldes &amp; Schneider). Whether these changes deliver the robustness and speed the paper called for is not systematically re-checked."),
           P("<b>Research questions</b>", ParagraphStyle("q", parent=BODY, fontName="Helvetica-Bold", textColor=RED)),
           B("<b>RQ1:</b> Does pooled-treebank training improve parsing robustness on unseen English genres compared to single-treebank models?"),
           B("<b>RQ2:</b> How large is the runtime gap to spaCy today, on CPU and GPU?"),
           P("<b>H1:</b> The pooled model has the best mean LAS on held-out treebanks (LinES, ParTUT), at a small cost on in-domain sets. <b>H2:</b> The gap narrows on GPU but remains &gt;1\u00d7 on CPU."),
           P("<b>Contribution:</b> a reproducible re-evaluation of L1 and L2 with the current Stanza release.")]
    meth = [P("2  Methodology", H2),
            P("<b>Models.</b> Stanza English packages <i>ewt</i> and <i>gum</i> (single treebank) and <i>combined</i> (pooled). Pipeline: tokenize, POS, lemma, depparse. Baseline: spaCy <i>en_core_web_sm</i> (NER disabled)."),
            P("<b>Data.</b> Official test splits of four UD English treebanks: EWT (web), GUM (multi-genre), LinES (parallel/novel), ParTUT (legal, Wikipedia, news). LinES and ParTUT are held out for all models (Stanza docs list training sources)."),
            P("<b>Protocol.</b> Gold tokens are given to isolate tagging and parsing from segmentation. Metrics: UPOS accuracy, UAS, LAS; dependency subtypes stripped as in the CoNLL-18 script."),
            P("<b>Speed.</b> First 1000 EWT test sentences as raw text; best of 2 runs after warm-up; tokens/s and runtime relative to spaCy CPU, as in the paper's Table 4."),
            P("<b>Why this design.</b> Comparing <i>ewt</i>/<i>gum</i>/<i>combined</i> on in-domain vs. held-out sets isolates the effect of pooling; holding pipeline and metric constant keeps the comparison fair."),
            P("<b>Reproducibility.</b> All code, data download and figure generation:"),
            P(f"<font name='Courier-Bold' color='#C1121F'>{a.repo}</font>"),
            P("Command: <font name='Courier'>python run_experiments.py</font>. Versions and hardware are stored in results.json.")]
    lay = [P("3  Results", H2)]
    fw, fh = 297, 105
    if res:
        fig_accuracy(res, fw, fh, "/tmp/fig_acc.png"); fig_speed(res, fw, 88, "/tmp/fig_speed.png")
        fa = Image("/tmp/fig_acc.png", width=fw * mm, height=fh * mm); fs = Image("/tmp/fig_speed.png", width=fw * mm, height=88 * mm)
    else:
        fa = Placeholder(fw * mm, fh * mm, "FIGURE 1 \u2013 run run_experiments.py"); fs = Placeholder(fw * mm, 88 * mm, "FIGURE 2 \u2013 run run_experiments.py")
    fnd, _ = findings(res)
    lay += [fa, P("Figure 1: LAS per model and test treebank (LinES, ParTUT held out).", CAP), results_table(res),
            P("Table 1: UPOS / LAS. Macro = mean over the four treebanks.", CAP), fs,
            P("Figure 2: Runtime relative to spaCy CPU (grey: values reported in 2020).", CAP)] + fnd + [
            P("<b>Limitations:</b> English only; one run (no seeds; models fixed); gold tokenization; speed depends on hardware and batch settings; 'combined' may include treebanks related to the test sets.")]
    frame_fill(c, xs[0], 15 * mm, cw[0], hh, hyp, "hypothesis")
    frame_fill(c, xs[1], 15 * mm, cw[1], hh, meth, "methodology")
    frame_fill(c, xs[2], 15 * mm, cw[2], hh, lay, "results")
    c.showPage(); c.save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors", default="<Your Name>")
    ap.add_argument("--repo", default="https://github.com/<user>/<repo>")
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--out", default="poster.pdf")
    build(ap.parse_args())
