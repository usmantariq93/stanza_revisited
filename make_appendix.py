"""Creates appendix.pdf: reference list + your SIGNED declaration.
   python make_appendix.py --declaration signed_declaration.pdf"""
import argparse, subprocess, shutil
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

REFS = [
 "Qi, P., Zhang, Y., Zhang, Y., Bolton, J., and Manning, C. D. (2020). Stanza: A Python Natural Language Processing Toolkit for Many Human Languages. In <i>Proceedings of the 58th Annual Meeting of the ACL: System Demonstrations</i>, pp. 101-108.",
 "Nivre, J., de Marneffe, M.-C., Ginter, F., Hajic, J., Manning, C. D., Pyysalo, S., Schuster, S., Tyers, F., and Zeman, D. (2020). Universal Dependencies v2: An Evergrowing Multilingual Treebank Collection. In <i>Proceedings of LREC 2020</i>.",
 "Silveira, N., Dozat, T., de Marneffe, M.-C., Bowman, S., Connor, M., Bauer, J., and Manning, C. D. (2014). A Gold Standard Dependency Corpus for English. In <i>Proceedings of LREC 2014</i>. (UD English-EWT)",
 "Zeldes, A. (2017). The GUM Corpus: Creating Multilayer Resources in the Classroom. <i>Language Resources and Evaluation</i>, 51(3), 581-612. (UD English-GUM)",
 "Dozat, T. and Manning, C. D. (2017). Deep Biaffine Attention for Neural Dependency Parsing. In <i>ICLR 2017</i>.",
 "Zeman, D. et al. (2018). CoNLL 2018 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies. In <i>Proceedings of the CoNLL 2018 Shared Task</i>. (evaluation protocol / LAS, UAS)",
 "Zeldes, A. and Schneider, N. <i>Are UD Treebanks Getting More Consistent? A Report Card for English UD.</i> Georgetown University. <b>[TODO: add venue and year from the paper]</b>",
 "Stanza documentation and release notes (v1.2.0: English default models trained on EWT+GUM): https://stanfordnlp.github.io/stanza/ and https://pypi.org/project/stanza/",
 "spaCy documentation and model card, en_core_web_sm: https://spacy.io/",
 "Universal Dependencies treebank documentation and data: https://universaldependencies.org/ ; UD_English-LinES, UD_English-ParTUT (test sets used).",
]
ap = argparse.ArgumentParser(); ap.add_argument("--declaration"); ap.add_argument("--out", default="appendix.pdf"); a = ap.parse_args()
st = getSampleStyleSheet(); doc = SimpleDocTemplate("_refs.pdf", pagesize=A4)
story = [Paragraph("Appendix", st["Title"]), Paragraph("A. References", st["Heading2"])]
for i, r in enumerate(REFS, 1): story += [Paragraph(f"[{i}] {r}", st["BodyText"]), Spacer(1, 6)]
story += [Spacer(1, 12), Paragraph("B. Declaration of Academic Integrity: signed form follows on the next page.", st["BodyText"])]
doc.build(story)
if a.declaration: subprocess.run(["qpdf", "--empty", "--pages", "_refs.pdf", a.declaration, "--", a.out], check=True)
else: shutil.copy("_refs.pdf", a.out); print("WARNING: no --declaration given; the appendix is INCOMPLETE without the signed form.")
