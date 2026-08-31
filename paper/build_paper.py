"""Assemble the single unified paper from the two working sources.

One paper, no parts: each topic is covered once, for both objectives. Section
numbers are assigned by position in `outline.OUTLINE`, and the old->new
cross-reference map is derived from the same table.

Output: paper.md, paper.tex (ICLR-style), paper.pdf.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import outline
import sections

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SRC = REPO.parent / "research" / "infinite_horizon_diversity"

TITLE = "Improving Diversity Metrics in Synthetic Data Generation"
SUBTITLE = ("One conditioning method, the metrics it can be pointed at, and "
            "the conditional dimension that limits all of them")

PREAMBLE = r"""
% ICLR-style page: 5.5in text block on US letter, Times, 10pt
\usepackage[letterpaper,left=1.5in,right=1.5in,top=1in,bottom=1in]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{microtype}
\usepackage[table]{xcolor}
\usepackage{caption}
\usepackage{titlesec}
\usepackage{titling}
\usepackage[colorlinks=true,linkcolor=blue!45!black,urlcolor=blue!45!black,
            citecolor=blue!45!black]{hyperref}
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}
\newtheorem{assumption}{Assumption}
\setlength{\emergencystretch}{3em}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\captionsetup{font=small,labelfont=bf}

\titleformat{\section}{\normalfont\large\bfseries}{\thesection}{0.7em}{}
\titleformat{\subsection}{\normalfont\normalsize\bfseries}{\thesubsection}{0.6em}{}
\titleformat{\subsubsection}{\normalfont\normalsize\bfseries\itshape}{\thesubsubsection}{0.6em}{}
\titlespacing*{\section}{0pt}{1.7ex plus .6ex minus .2ex}{1.0ex plus .2ex}
\titlespacing*{\subsection}{0pt}{1.3ex plus .5ex minus .2ex}{0.8ex plus .2ex}
\titlespacing*{\subsubsection}{0pt}{1.1ex plus .4ex minus .2ex}{0.7ex plus .2ex}

\pretitle{\begin{center}\LARGE\bfseries}
\posttitle{\par\end{center}\vskip 0.4em}
\preauthor{\begin{center}\large}
\postauthor{\par\end{center}}
\predate{\begin{center}\normalsize}
\postdate{\par\end{center}\vskip 0.8em}

\newenvironment{iclrabstract}
  {\vspace{0.4em}\begin{center}{\bfseries\scshape Abstract}\end{center}
   \begin{list}{}{\setlength{\leftmargin}{0.45in}\setlength{\rightmargin}{0.45in}}\item[]\relax}
  {\end{list}\vspace{0.6em}}

% wide tables and figures shrink to the measure instead of running off the page
\newcommand{\fitwidth}[1]{\makebox[\linewidth][c]{\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{#1}}}

% Times carries no mathematical glyphs; give those code points a real font
\usepackage{newunicodechar}
\newfontfamily\symfont{STIX Two Math}[Scale=MatchLowercase]
\newunicodechar{ℝ}{{\symfont ℝ}}
\newunicodechar{ℙ}{{\symfont ℙ}}
\newunicodechar{𝔼}{{\symfont 𝔼}}
\newunicodechar{∈}{{\symfont ∈}}
\newunicodechar{⋃}{{\symfont ⋃}}
\newunicodechar{∪}{{\symfont ∪}}
\newunicodechar{⊆}{{\symfont ⊆}}
\newunicodechar{⊂}{{\symfont ⊂}}
\newunicodechar{∝}{{\symfont ∝}}
\newunicodechar{∎}{{\symfont ∎}}
\newunicodechar{≪}{{\symfont ≪}}
\newunicodechar{∼}{{\symfont ∼}}
\newunicodechar{∅}{{\symfont ∅}}
\newunicodechar{₉}{{\symfont ₉}}
\newunicodechar{₅}{{\symfont ₅}}
\newunicodechar{⁻}{{\symfont ⁻}}

\pagestyle{plain}
\setlength{\parskip}{0.45em}
\setlength{\parindent}{0pt}
"""

ABSTRACT = """
A synthetic corpus is built to be good at something measurable, and the measure
is chosen before the corpus is. **Coverage** asks the corpus to reach as much of
the space as possible in *n* generator calls. **Max-min** asks that no two of the
*n* items resemble each other. Duplicate rate, mean-centered Vendi, precision
against a reference and worst-case nearest-neighbour distance are all in use, and
a corpus that scores well on one can score badly on another. This paper is about
how to build a generator loop that improves such a measure, and about how much of
that loop has to change when the measure does.

The method is **Recursive Axis Conditioning** (RAC). We assume an embedding oracle
and, critically, **no inverse**: we can compute exactly where the next item ought
to land and have no way to decode that point into text. Every architectural choice
follows. RAC asks the generator to name the axes along which its own outputs can
differ, ranks those axes, selects their most-different values, and splits an axis
with nothing new left to offer into finer sub-axes that apply only inside the
region that exhausted it. We study two measures in depth, coverage and max-min,
and find that the loop is shared and the measure enters at two points: how a
candidate axis is scored, and how one of K candidates is selected. Those two
points are enough to reverse the ranking of methods, so a measure has to be named
before a diversity number means anything.

No measure here is limited by its own optimizer. All of them are limited by the
**support**: conditioned on a fixed prompt, a language model's output concentrates
on a submanifold of dimension *m* far below the dimension *d* of the space it
could reach. Fifteen numerical checks establish the consequences. Novelty at fixed
prompt decays as $n^{-1/m}$, not $n^{-1/d}$; one prompt ε-covers a vanishing
$ε^{d-m}$ fraction; only prompt motion *transverse* to the already-occupied span
raises the ceiling; and the optimal number of samples per prompt is set by the
ratio of prompt-switching cost to sampling cost.

On ~46,000 real generations from `openai/gpt-5.6-luna`, ~700 rendered images and
~200 rendered instrumentals, RAC places first among twelve corpora against the
released Alpaca, PersonaHub and WizardLM sets — 0.4441 against Alpaca's 0.3722 at
matched evaluated-*n* on a human-written reference no corpus was aimed at, on one
twentieth of Alpaca's generation budget — and wins every literal and latent
measure against five published methods under the max-min objective. Along the way:
a psychometric corpus that is 73.6% exact duplicates with one item repeated 2,726
times, which temperature barely dents (71.0%) and conditioning nearly eliminates
(0.0%); literal and latent diversity moving in opposite directions as *n* grows;
and text-embedding similarity predicting rendered-image similarity at only
***r* = 0.170**. The two measures also resist being served by one tool: applying
the max-min side's orthogonalized conditioning to the coverage objective scores
below plain conditioning, because steering away from the occupied span steers away
from where the reference measure is densest.
"""


# sections absorbed into another, rather than kept as a block of their own
EXTRA_REMAP = {
    "mm": {"1.2": "2.2", "3": "5", "4": "6", "7": "4", "6": "7"},
    "cv": {"1": "2", "3": "5", "9": "4", "6": "7.12", "7.1": "5.11", "7.2": "5.11",
           "7.3": "5.11", "7.4": "5.11", "7.5": "5.11", "7.6": "5.11", "7.7": "5.11",
           "7.8": "5.11"},
    "iii": {},
}



# figure filenames referenced by "Figure N" inside each source, so the merged
# paper can renumber every figure by order of appearance
FIG_BY_SOURCE = {
    "mm": {1: "fig5_scaling.png", 2: "fig8_breadth_depth.png", 3: "fig15_rac_diagram.png",
           4: "fig10_real_curves.png", 5: "fig11_literal_vs_latent.png",
           6: "fig14_arms.png", 7: "fig12_vision.png", 8: "fig13_contact_sheet.png"},
    "cv": {3: "fig15_rac_diagram.png", 4: "fig_allocation.png",
           6: "fig_live_pilot.png", 7: "fig_literal_vs_latent.png"},
    "iii": {},
}
FIGREF = re.compile(r"(?:Part I's )?Figure (\d+)")
UNNUMBERED = {  # coverage captions that never carried a number
    "fig_h2h_scalefree.png", "fig_benchmark.png", "fig_live_pilot.png",
    "fig_allocation.png",
}


def tokenize_figs(text: str, kind: str) -> str:
    """Replace source-local figure numbers with filename tokens."""
    table = FIG_BY_SOURCE.get(kind, {})
    def sub(m):
        f = table.get(int(m.group(1)))
        return "{{FIG:" + f + "}}" if f else m.group(0)
    text = FIGREF.sub(sub, text)
    # give the unnumbered coverage captions a token of their own
    def cap(m):
        name = m.group(2).split("/")[-1]
        if name in UNNUMBERED and "{{FIG:" not in m.group(1):
            return f"![{{{{FIG:{name}}}}}. {m.group(1)}]({m.group(2)})"
        return m.group(0)
    return re.sub(r"!\[([^\]]+)\]\(([^)]+)\)", cap, text)


def number_figures(md: str) -> str:
    """Assign figure numbers by order of appearance and resolve every token."""
    order, seen = {}, 0
    for m in re.finditer(r"!\[\{\{FIG:([^}]+)\}\}", md):
        name = m.group(1)
        if name not in order:
            seen += 1
            order[name] = seen
    md = re.sub(r"\{\{FIG:([^}]+)\}\}",
                lambda m: f"Figure {order.get(m.group(1), '?')}", md)

    # the italic paragraph under an image mirrors its caption; keep the label
    def mirror(m):
        num, cap, rest = m.group(1), m.group(2), m.group(3)
        if rest.strip() == cap.strip():
            head = m.group(0)[:m.start(3) - m.start(0)]
            return f"{head}Figure {num}. {rest}*"
        return m.group(0)
    md = re.sub(r"!\[Figure (\d+)\. ([^\]]+)\]\([^)]+\)\n\n\*([^*]+)\*",
                mirror, md)
    return md



# the sources were written as two papers; in one paper the framing has to go
SEAMS = [
    ("**Coverage.** Part II treats the covering objective",
     "**Coverage.** The covering objective is treated below"),
    ("the packing objective Part I\nstreams", "the packing objective this paper streams"),
    ("trick as in Part I.", "trick as in the packing objective."),
    ("the greedy algorithm for the k-center / packing objective, and Part I",
     "the greedy algorithm for the k-center / packing objective, and this paper"),
    ("Part II's \u00a7II.5 compares RAC-coverage", "\u00a77 compares RAC-coverage"),
    ("The configuration applying Part I's\northogonalized conditioning",
     "The configuration applying the packing side's\northogonalized conditioning"),
    ("Part I \u00a7I.3. Both need", "\u00a75. Both need"),
    ("**Depth per condition and switch cost.** Part I's slice theory sharpens the",
     "**Depth per condition and switch cost.** The slice theory of \u00a75 sharpens the"),
    ("At that switch cost Part I's", "At that switch cost the"),
    ("**7.7 Lattice size is not reachable dimension.** Using Part I's",
     "**Lattice size is not reachable dimension.** Using the"),
    ("**7.8 Refinement direction: density vs reach.** Part I's negative",
     "**Refinement direction: density versus reach.** The earlier"),
    ("The pipeline is Part I's **Recursive Axis Conditioning** (RAC) stack with the selection\n"
     "objective and its bookkeeping swapped from packing to covering; we write RAC-coverage for\n"
     "this instantiation and RAC-packing for Part I's.",
     "The covering instantiation is the same RAC stack with the selection objective\n"
     "and its bookkeeping swapped from packing to covering; we write RAC-coverage for\n"
     "it and RAC-packing for the objective of the preceding sections."),
    ("**Score-guided conditioning (the coverage adaptation).** Part I's generation\naxis scoring",
     "**Score-guided conditioning (the coverage adaptation).** The generation\naxis scoring of \u00a76.3"),
    ("mass. We keep Part I's embedding-based spread and", "mass. We keep the embedding-based spread and"),
    ("**Attractor ledger.** Every 20 accepted items (50 in Part I), the accepted",
     "**Attractor ledger.** Every 20 accepted items (50 under the packing objective), the accepted"),
    ("conditional axes forming a tree, exactly as in\nPart I.", "conditional axes forming a tree, as above."),
    ("min-gap metric no budgeted product requirement asks for. Part I's stack",
     "min-gap metric no budgeted product requirement asks for. The packing stack"),
    ("the packing objective Part I\npaper streams", "the packing objective this paper streams"),
    ("Part II measured a mean pairwise cosine", "\u00a77.12 measures a mean pairwise cosine"),
    ("consistent with Part III: coverage is monotone",
     "consistent with \u00a74: coverage is monotone"),
    ("Part II's objective is the one that has a reference distribution in it",
     "The covering objective is the one that has a reference distribution in it"),
    ("We call the method **Recursive Axis Conditioning** (RAC): the generator is asked for "
     "the language-valued axes along which its own outputs can differ, those axes are ranked "
     "and their most-different levels chosen, and an axis that runs out of transverse variation "
     "is split into finer conditional sub-axes.",
     "The loop below is written once and pointed at a measure. We call it **Recursive Axis "
     "Conditioning** (RAC): the generator is asked for the language-valued axes along which its "
     "own outputs can differ, those axes are ranked and their most-different levels chosen, and "
     "an axis that runs out of transverse variation is split into finer conditional sub-axes. "
     "Two components read the measure and the rest do not, which is what makes the same stack "
     "serve objectives whose optima conflict."),
    ("Steps 1\u20137 are shared by both objectives. The objective enters at two points only:",
     "Steps 1\u20137 are shared whatever the measure. The measure enters at two points only:"),
    ("Generator: `openai/gpt-5.6-luna` via OpenRouter. Text embeddings:",
     "Every number below states the measure it is computed under and the *n* it is computed at, "
     "since \u00a74.1 showed how far the measures can diverge on one corpus. Generator: "
     "`openai/gpt-5.6-luna` via OpenRouter. Text embeddings:"),
]


def fix_seams(md: str) -> str:
    for old, new in SEAMS:
        md = md.replace(old, new)
    return md




# plain-text set notation from the working drafts, promoted to real maths so it
# typesets instead of running off the measure as one unbreakable token
MATHFIX = [
    ("F_μ,ε(S) = μ( ⋃_{x∈S} B(x, ε) ),",
     r"$F_{\mu,\varepsilon}(S) = \mu\!\left(\bigcup_{x \in S} B(x,\varepsilon)\right)$,"),
    ("F(S) = μ(⋃_{x∈S} B(x, ε))",
     r"$F(S) = \mu(\bigcup_{x \in S} B(x,\varepsilon))$"),
    ("for S ⊆ T and any x,", r"for $S \subseteq T$ and any $x$,"),
    ("The marginal gain of x given S is μ(B(x, ε) \\ ⋃_{y∈S}\nB(y, ε)).",
     r"The marginal gain of $x$ given $S$ is $\mu(B(x,\varepsilon) \setminus \bigcup_{y \in S} B(y,\varepsilon))$."),
    ("Since S ⊆ T implies ⋃_S B(y,ε) ⊆ ⋃_T B(y,ε), the set being",
     r"Since $S \subseteq T$ implies $\bigcup_S B(y,\varepsilon) \subseteq \bigcup_T B(y,\varepsilon)$, the set being"),
    ("N_ε(x) = {p : ‖p − x‖ ≤ ε} of the pool, and F̂(S) = |⋃_{x∈S} N_ε(x)|/P) so",
     r"$N_\varepsilon(x) = \{p : \lVert p - x\rVert \le \varepsilon\}$ of the pool, and $\hat F(S) = |\bigcup_{x \in S} N_\varepsilon(x)|/P$) so"),
    ("F̂(S_greedy) ≥ (1 − 1/e) · max_{|S|≤n} F̂(S) ≈ 0.632 · OPT,",
     r"$\hat F(S_{\text{greedy}}) \ge (1 - 1/e) \cdot \max_{|S| \le n} \hat F(S) \approx 0.632 \cdot \mathrm{OPT}$,"),
    ("E_{x∼axis}[F(S ∪ {x}) − F(S)]",
     r"$\mathbb{E}_{x \sim a}[F(S \cup \{x\}) - F(S)]$"),
    ("dim(⋃<sub>*j*</sub> *S<sub>j</sub>*) = min(*d*, *m* + rank{*c<sub>j</sub>* − "
     "*c*<sub>1</sub>}), where *c<sub>j</sub>* is the centre of *S<sub>j</sub>*. In "
     "particular, displacements lying inside span(*S*<sub>1</sub>) contribute nothing "
     "to the union's dimension.",
     r"$\dim\!\left(\bigcup_j S_j\right) = \min\!\left(d,\; m + "
     r"\operatorname{rank}\{c_j - c_1\}\right)$, where $c_j$ is the centre of "
     r"$S_j$. In particular, displacements lying inside $\operatorname{span}(S_1)$ "
     r"contribute nothing to the union's dimension."),
]


def fix_math(md: str) -> str:
    for old, new in MATHFIX:
        md = md.replace(old, new)
    return md


def assemble():
    """Two passes: the first fixes section numbers, the second rewrites the
    cross-references inside each block against the map of its own source."""
    mm = sections.parse((SRC / "PAPER.md").read_text())
    cv = sections.parse((SRC / "coverage" / "PAPER.md").read_text())
    from part_three import PART_THREE
    iii = sections.parse(PART_THREE)
    src = {"mm": mm, "cv": cv, "iii": iii}

    # this figure is declared inside the coverage paper's related-work section;
    # in one paper it belongs beside the results it reports
    bench = ("![Selection benchmark: coverage-greedy against five literature "
             "baselines.](figures/fig_benchmark.png)\n\n*Selection benchmark: "
             "coverage-greedy against five literature baselines.*\n")
    cv["8"]["body"] = cv["8"]["body"].replace(bench, "")
    cv["5.3"]["body"] = cv["5.3"]["body"].rstrip() + "\n\n" + bench

    chunks, remap = [], {"mm": {}, "cv": {}, "iii": {}}
    for si, (stitle, items) in enumerate(outline.OUTLINE, start=1):
        chunks.append((None, f"## {si}. {stitle}\n"))
        sub_i = 0
        for item in items:
            if item[0] == "text":
                chunks.append((None, item[1].strip() + "\n"))
                continue
            if item[0] == "sub":
                _, title, kind, key = item
                sub_i += 1
                num = f"{si}.{sub_i}"
                chunks.append((None, f"### {num} {title}\n"))
                if kind == "text":
                    chunks.append((None, key.strip() + "\n"))
                    continue
                chunks.append((kind, sections.body(src[kind], key).rstrip("-\n ")))
                remap[kind][key] = num
            else:
                kind, key = item
                chunks.append((kind, sections.body(src[kind], key).rstrip("-\n ")))
                remap[kind][key] = str(si)

    full = {k: {**remap[k], **EXTRA_REMAP.get(k, {})} for k in remap}
    body = []
    for kind, text in chunks:
        body.append(tokenize_figs(rewrite_refs(text, full[kind]), kind)
                    if kind else text)

    head = ["## Abstract", "", ABSTRACT.strip(), "", "---", ""]
    md = "\n".join(head) + "\n" + "\n\n".join(body) + "\n"
    md = re.sub(r"<!--FIG:[^>]*-->\n?", "", md)   # invisible in output anyway
    return fix_math(fix_seams(number_figures(md)))


def rewrite_refs(text: str, mapping: dict) -> str:
    """Point every § reference in one block at its new section number."""
    def sub(m):
        key = m.group(1).rstrip(".")
        tail = m.group(1)[len(key):]
        return "\u00a7" + mapping.get(key, key) + tail
    return re.sub(r"\u00a7([0-9]+(?:\.[0-9]+)*[a-z]?\.?)", sub, text)






LT = re.compile(r"\\begin\{longtable\}\[\]\{@\{\}([lrc]+)@\{\}\}(.*?)\\end\{longtable\}", re.S)


def fit_tables(tex: str) -> str:
    """Give unsized tables wrapping columns so no row runs into the margin.

    Pandoc emits `@{}ll@{}` for any table whose source cells are short, and a
    single long cell then pushes the whole row off the page. Widths are set in
    proportion to each column's longest cell, damped so one long column does
    not starve the rest.
    """
    def one(m):
        spec, body = m.group(1), m.group(2)
        n = len(spec)
        widths = [1] * n
        for row in body.split(r"\\"):
            if "&" not in row:
                continue
            for i, cell in enumerate(row.split("&")[:n]):
                widths[i] = max(widths[i], len(cell.strip()))
        w = [v ** 0.5 for v in widths]
        tot = sum(w)
        w = [x / tot for x in w]
        col = r">{\raggedright\arraybackslash}p{\dimexpr %.4f\linewidth-2\tabcolsep\relax}"
        cols = "".join(col % x for x in w)
        size = r"\footnotesize" if (n >= 4 or max(widths) > 45) else r"\small"
        return ("{" + size + "\n" + r"\begin{longtable}[]{@{}" + cols + r"@{}}"
                + body + r"\end{longtable}}")
    return LT.sub(one, tex)


def iclr_polish(tex: Path) -> None:
    """Pandoc's numbered Abstract section becomes the ICLR abstract block."""
    t = tex.read_text()
    t = re.sub(r"\\(?:sub)*section\{Abstract\}\\label\{abstract\}",
               r"\\begin{iclrabstract}", t, count=1)
    m = re.search(r"\\begin\{iclrabstract\}", t)
    if m:
        nxt = min((i for i in (t.find(f"\\{k}section{{", m.end())
                               for k in ("", "sub", "subsub")) if i > 0), default=-1)
        if nxt > 0:
            t = t[:nxt] + "\\end{iclrabstract}\n\n" + t[nxt:]
    t = fit_tables(t)
    tex.write_text(t)


def build_pdf() -> None:
    figdir = HERE / "figures"
    figdir.mkdir(exist_ok=True)
    n = 0
    for src_dir in (SRC / "figures", SRC / "coverage" / "figures"):
        for f in src_dir.glob("*.png"):
            shutil.copy(f, figdir / f.name)
            n += 1
    print(f"copied {n} figures")
    pre = HERE / ".preamble.tex"
    pre.write_text(PREAMBLE)
    subprocess.run([
        "pandoc", str(HERE / "paper.md"), "-f", "gfm+tex_math_dollars",
        "-t", "latex", "-s", "--pdf-engine=xelatex",
        "-V", "documentclass=article", "-V", "fontsize=10pt",
        "-V", "mainfont=Times New Roman", "-V", "monofont=Menlo",
        "-M", f"title={TITLE}", "-M", f"subtitle={SUBTITLE}", "-M", "date=",
        "-H", str(pre), "-o", str(HERE / "paper.tex")], check=True, cwd=HERE)
    iclr_polish(HERE / "paper.tex")
    print("paper.tex written")
    for _ in range(2):
        subprocess.run(["xelatex", "-interaction=nonstopmode", "paper.tex"],
                       cwd=HERE, capture_output=True, text=True)
    pdf = HERE / "paper.pdf"
    if pdf.exists():
        print(f"paper.pdf written ({pdf.stat().st_size // 1024} KB)")
    pre.unlink(missing_ok=True)


def main():
    md = assemble()
    (HERE / "paper.md").write_text(md)
    print(f"paper.md written ({len(md.split())} words)")
    build_pdf()


if __name__ == "__main__":
    main()
