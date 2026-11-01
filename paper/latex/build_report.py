"""Builds report.tex: the SAFL submission template's sections, filled in, with figures from main.tex."""
import re, sys
D = sys.argv[1]
src = open(D + "/main.tex", encoding="utf-8").read()
pre = src[:src.index(r"\begin{document}")]
pre = pre.replace(r"\documentclass[10pt,twocolumn]{article}", r"\documentclass[11pt]{article}")
pre = pre.replace("left=0.62in,right=0.62in,columnsep=0.26in", "left=0.8in,right=0.8in").replace("bottom=0.8in", "bottom=0.7in")
pre = pre.replace(r"\usepackage{flushend}", "") + "\\usepackage{tabularx}\n\\usepackage{multicol}\n\\usepackage{xltabular}\n\\renewcommand{\\topfraction}{0.9}\n\\renewcommand{\\bottomfraction}{0.8}\n\\renewcommand{\\floatpagefraction}{0.75}\n\\setcounter{totalnumber}{4}\n\\setlength{\\textfloatsep}{10pt plus 2pt minus 2pt}\n\\setlength{\\floatsep}{8pt plus 2pt minus 2pt}\n\\renewcommand{\\arraystretch}{1.15}\n"
pre = pre.replace(r"\fancyhead[L]{\footnotesize\color{sub}Passing Isn't Proof}", r"\fancyhead[L]{\footnotesize\color{sub}Epoch Zero · Passing Isn't Proof}")
pre = pre.replace(r"\renewcommand{\textfraction}{0.05}", r"\renewcommand{\textfraction}{0.08}")

pre = pre.replace(r"\usepackage[round,authoryear]{natbib}", r"\usepackage[numbers,square,sort&compress]{natbib}")

def block(label):
    k = src.index(r"\label{" + label + "}")
    env = "table" if label.startswith("tab") else "figure"
    i = max(src.rfind(r"\begin{" + env + "*}", 0, k), src.rfind(r"\begin{" + env + "}", 0, k))
    j = src.index(r"\end{" + env, k); j = src.index("}", j) + 1
    b = src[i:j].replace(env + "*}", env + "}")
    b = re.sub(r"\\begin\{" + env + r"\}\[[^\]]*\]", r"\\begin{" + env + "}[!tbp]", b)
    if not b.startswith(r"\begin{" + env + "}["): b = b.replace(r"\begin{" + env + "}", r"\begin{" + env + "}[!tbp]", 1)
    cut = r"\par\vspace{2.5em}"
    if cut in b: b = b[:b.index(cut)] + r"\end{figure}"
    b = b.replace(r"\columnwidth", r"\textwidth")
    # every figure at full text width: natural aspect ratio, nothing squeezed
    return re.sub(r"\\fig\[[0-9.]*\\textwidth\]", lambda m: r"\fig[\textwidth]", b)

def tasks_table():
    k = src.index(r"\captionof{table}")
    i = src.rindex(r"\resizebox", 0, k); j = src.index(r"\label{tab:tasks}", k) + len(r"\label{tab:tasks}")
    body = src[i:j].replace(r"\captionof{table}", r"\caption")
    return "\\begin{table}[!tbp]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{4pt}\n" + body + "\n\\end{table}\n"

bib = src[src.index(r"\begin{thebibliography}"):src.index(r"\end{thebibliography}") + len(r"\end{thebibliography}")]

def fields(rows, w="0.34"):
    out = "\\noindent\\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}p{" + w + "\\textwidth} >{\\raggedright\\arraybackslash}X@{}}\n\\toprule\n\\textbf{Field} & \\textbf{Your answer} \\\\\n\\midrule\n"
    out += "\n".join(f"{a} & {b} \\\\" for a, b in rows)
    return out + "\n\\bottomrule\n\\end{tabularx}\n"

def q(title, body): return f"\\paragraph{{{title}}}\\mbox{{}}\\\\\n{body}\n"

T = open(D + "/template_body.tex", encoding="utf-8").read()
for lab in re.findall(r"%%FIG:(\S+)", T): T = T.replace("%%FIG:" + lab, block(lab))
# PRUNE: keep only references cited in the report
cited = set(k.strip() for grp in re.findall(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}", T) for k in grp.split(","))
items = re.split(r"(?=\\bibitem)", bib)
END = "\\end{thebibliography}"
head, tail = items[0], items[-1][items[-1].index(END):]
items[-1] = items[-1][:items[-1].index(END)]
key = lambda it: re.search(r"\\bibitem\[[^\]]*\]\{([^}]*)\}", it).group(1)
kept = [it for it in items[1:] if key(it) in cited]
missing = cited - {key(it) for it in items[1:]}
print("cited", len(cited), "kept", len(kept), "missing", missing)
bib = head.replace(r"\footnotesize", r"\scriptsize\setlength{\bibsep}{1pt}\raggedright\interlinepenalty=10000") + "".join(kept) + tail
bib = "\\section*{References}\n\\renewcommand{\\bibsection}{}\n\\begin{multicols}{2}\n" + bib + "\n\\end{multicols}\n"
T = T.replace("%%TASKS", tasks_table()).replace("%%BIB", bib)
open(D + "/report.tex", "w", encoding="utf-8").write(pre + "\\begin{document}\n" + T + "\n\\end{document}\n")
print("ok")
