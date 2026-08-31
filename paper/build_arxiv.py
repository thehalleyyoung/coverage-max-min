"""
Assemble the single arXiv submission from the two papers.

The two halves were written separately and share a theory section, a axis scoring,
a generator, an embedder stack and several findings. Concatenating them would
repeat all of that. So this script builds ONE paper with a shared front matter
and theory, then the two objectives as parallel parts, then a joint discussion
of where they disagree -- which is the actual contribution of putting them
together.

Output: paper.tex (self-contained), paper.pdf (via xelatex), paper.md.
Figures are copied next to the .tex so the submission tarball is complete.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SRC_MAXMIN = REPO.parent / "research" / "infinite_horizon_diversity"
SRC_COV = SRC_MAXMIN / "coverage"

PREAMBLE = r"""
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{microtype}
\usepackage[table]{xcolor}
\usepackage{caption}
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
\captionsetup{font=small}
"""


def strip_front(md: str) -> str:
    """Drop a paper's own title/abstract block; the merged paper supplies one."""
    i = md.find("\n## ")
    return md[i:] if i > 0 else md


def demote(md: str, levels: int = 1) -> str:
    """Push every heading down so the merged paper's parts sit above them."""
    out = []
    for line in md.split("\n"):
        m = re.match(r"^(#{1,6})\s", line)
        out.append(("#" * levels) + line if m else line)
    return "\n".join(out)


# base may be a Latin letter, a digit, a Greek/blackboard symbol, or a
# closing paren from an expression like (1 - p)<sup>K</sup>
SUP = re.compile(r"(\*?[A-Za-z0-9]\*?|[^\sA-Za-z0-9*<>])<sup>(.+?)</sup>")


def fix_superscripts(md: str) -> str:
    """Turn `*n*<sup>-1/*m*</sup>` into real math.

    Pandoc renders the HTML tag as a raw baseline string, so the exponent
    reads as a subtraction: "n-1/m" instead of n^{-1/m}. In a paper whose
    central claim IS an exponent, that is not a cosmetic problem.
    """
    def repl(m):
        base, exp = m.group(1), m.group(2)
        clean = lambda t: (t.replace("*", "").replace("\u2212", "-")
                           .replace("\u2013", "-"))
        base, exp = clean(base), clean(exp)
        # a closing paren belongs to the expression, not the exponent base --
        # emit it outside the math so the paren pairing survives
        return f"${base}^{{{exp}}}$"
    return SUP.sub(repl, md)


def renumber(md: str, prefix: str) -> str:
    """Prefix section numbers so the two parts don't both call themselves §3."""
    md = re.sub(r"^(#{2,6})\s+(\d+)(\.\d+)?\.?\s+",
                lambda m: f"{m.group(1)} {prefix}{m.group(2)}{m.group(3) or ''} ",
                md, flags=re.M)
    # inline cross-references belong to the part that wrote them
    md = re.sub(r"§(\d+(?:\.\d+)*)", lambda m: f"\u00a7{prefix}{m.group(1)}", md)
    # bolded subsection labels inside a run-in section (e.g. **7.1 Submodularity**)
    md = re.sub(r"\*\*(\d+\.\d+)( [A-Z])",
                lambda m: f"**{prefix}{m.group(1)}{m.group(2)}", md)
    return md


HEAD = r"""# Coverage and Max-Min Diversity in Synthetic Data Generation

**Two objectives, one budget, and the conditional dimension that limits both**

---

## Abstract

Synthetic corpora are generated under a fixed budget of *n* model calls, and two
different things are wanted from that budget. **Coverage** asks to reach as much
of the space as possible in *n* turns. **Max-min** asks that no two of the *n*
items resemble each other. They pull in different directions: coverage will
place two items near each other if between them they reach a large region, and
max-min will leave most of the space empty provided nothing collides. We study
both, on one generator, one set of embedders, and one budget, and we show they
disagree sharply enough that a method optimizing either can be near-worst at the
other — greedy k-center wins min-gap in both our domains and finishes **last** on
coverage, at a sixth of random.

Our central claim is that neither objective is limited by its own optimizer. Both
are limited by the **support**. Conditioned on a fixed prompt, a language model's
output concentrates on a submanifold of dimension *m* far below the dimension *d*
of the space it could reach, and fifteen numerical checks establish the
consequences: novelty at fixed prompt decays as *n*<sup>−1/*m*</sup>, not
*n*<sup>−1/*d*</sup>; one prompt ε-covers a vanishing ε<sup>*d*−*m*</sup>
fraction; only prompt motion *transverse* to the already-occupied span raises the
ceiling; and the optimal number of samples to draw per prompt is set by the ratio
of prompt-switching cost to sampling cost, reaching one when switching is free.

We assume an embedding oracle and, critically, **no inverse**. We can compute
exactly where the next item ought to land and have no way to decode that point
into text. Every architectural choice follows: the system must propose, measure
and select rather than solve, and its only steering handles are language-valued.
The method we build from that constraint is **Recursive Axis Conditioning**
(RAC). The generator is asked to name the axes along which its own outputs can
differ; those axes are ranked by an **axis-scoring rule** (spread ×
transversality × independence × headroom) their most-different values are
selected, and an axis with nothing transverse left to offer is split into finer
sub-axes that apply only inside the region that exhausted it. The score takes two
forms, one per objective, differing on exactly the two terms the objectives
differ on, and it is evaluated in the space the artifact actually occupies rather
than only in text.

We validate on ~46,000 real generations from `openai/gpt-5.6-luna`, ~700 rendered
images, and ~200 rendered instrumentals, measuring diversity at three levels:
literal (*n*-gram), latent (text embedding), and (where the artifact is not text) in the space the product occupies (CLIP for images, CLAP and MERT for audio).
On coverage, RAC places first among twelve corpora against the released Alpaca,
PersonaHub and WizardLM sets — 0.4441 against Alpaca's 0.3722 at matched
evaluated-*n* on a human-written reference no corpus was aimed at, on
one-twentieth of Alpaca's generation budget and with the highest precision in the
field. On max-min, it wins every literal and latent measure against five
published methods in both text domains, reaching 124.8 mean-centered Vendi on
psychometric items where persona conditioning reaches 99.7 and naive prompting
7.1. Along the way: a psychometric corpus that is **72.3% exact duplicates** with
one item repeated 1,637 times, which temperature barely dents (61.6%) and
conditioning nearly eliminates (0.2%); literal and latent diversity moving in
**opposite directions** as *n* grows; and text-embedding similarity predicting
rendered-image similarity at only ***r* = 0.170**, so every text-side method
here optimizes a proxy that explains about 3% of the variance in what the reader
receives. The two objectives resist being served by one tool: the arm applying
Part I's orthogonalized conditioning to the coverage objective scores below plain
conditioning, because steering away from the occupied span steers away from
where the reference measure is densest.

---
"""

JOINT = r"""
---

# Part III — Where the two objectives disagree

Everything above was two papers sharing a theory section. This part is the reason
to publish them together: run both objectives on the same generator, the same
embedder and the same budget, and they rank methods almost
oppositely.

## III.1 The dissociation, measured

In a leakage-free selection benchmark (one shared candidate pool, budget matched
at 700 and 1,896 generations respectively, selector shown only an estimation half
of the reference and scored on a disjoint held-out half):

| selector | DALL·E coverage | psychometric coverage | DALL·E min-gap |
|---|---|---|---|
| coverage-greedy (ours) | **0.4829** | **0.3901** | 0.299 |
| coverage-stream (ours) | 0.4200 | 0.3785 | 0.283 |
| ROUGE-L filter (Self-Instruct) | 0.4629 | 0.3575 | 0.208 |
| SemDeDup | 0.4600 | 0.3575 | 0.336 |
| random | 0.4629 | 0.3533 | 0.198 |
| MAP-DPP greedy | 0.3829 | 0.0988 | 0.402 |
| k-center (Gonzalez) | 0.4114 | 0.0557 | **0.491** |

**k-center wins min-gap outright in both domains and finishes last on coverage**,
at 0.0557 against random's 0.3533 in the psychometric domain — a sixth of what
random selection achieves. The two highest-Vendi selectors (k-center and MAP-DPP)
are the two worst covering ones. A practitioner who reads "diversity" off a Vendi
score and deploys the selector that maximizes it will get a corpus that covers
less of the space than picking at random.

## III.2 Why the scoring rule has to fork

The four factors are shared but two of them invert:

| | max-min / packing | coverage / covering |
|---|---|---|
| the ask | no two items alike, in *n* turns | reach as much as possible, in *n* turns |
| governed by | the closest pair | the bulk |
| submodular | no | yes, so greedy carries (1 − 1/e) |
| classical kin | k-center | facility location |
| **spread(a)** | **min** distance between level centroids | **measure-weighted mean** distance |
| **headroom(a)** | levels not yet **used** | measure not yet **covered** |
| value selection | `farthest_levels` (max-min subset) | `coverage_levels` (greedy marginal gain) |
| failure mode | chases outliers into off-manifold junk | leaves the frontier empty |

The behavioural difference is visible on a two-line test. Given an axis with three
tightly-clustered common levels and one rare outlying level, `farthest_levels`
picks the outlier **first** and `coverage_levels` picks a cluster centre first and
the outlier second. That is k-center versus facility location, reproduced inside
the axis axis scoring, and it is the mechanism behind the table above.

## III.2b Are these dual problems?

**Where they are dual.** For radius ε, a *maximal* ε-packing is automatically an
ε-covering: any uncovered point could have been added to the packing. We confirm
this numerically on 3,000 held-out human instructions — at every ε tested, the
greedy maximal packing covers 100.0% of the set, and the classical sandwich
N_cov(ε) ≤ N_pack(ε) ≤ N_cov(ε/2) holds. Max-min run to maximality *is* a
covering algorithm, and greedy k-center 2-approximates the covering **radius**.

**Where they are not.** That duality concerns the worst-case radius: the distance
from the least-covered point to its nearest center. The coverage that matters
for synthetic data is **measure-weighted** — what fraction of the reference
distribution lies within reach, and the two come apart exactly when the measure
is non-uniform, which it always is. k-center is driven by outliers: every
isolated point sets the max and so commands a center. Measure-weighted coverage
is driven by mass: an outlier is worth its own measure and no more. At a budget
far below what the space needs, the prescriptions are opposite, and that is the
entire content of our k-center result (first on min-gap, last on coverage). The
true dual of measure-weighted coverage is fractional set cover, whose primal is
submodular; max-min is not.

**What the duality still buys: packing calibrates, covering places.** Even where
packing is the wrong way to *place* points, it is the right way to *choose the
scale*. Define ε\*(n) as the radius at which a maximal packing of the reference
has about n points — the finest resolution an n-item corpus can deliver.
Measured on the held-out human reference: ε\*(100) = 0.549, ε\*(450) = 0.472,
ε\*(1000) = 0.418. This retroactively diagnoses our failed fixed-ε comparison
with more precision than "the ε was too small": its ε sat at the 2nd percentile
of pairwise distances *regardless of budget*, while a 450-item corpus genuinely
supports about the 2.6th percentile — the error was choosing resolution
independently of n. Protocol: calibrate ε by packing, then place by greedy
marginal coverage against the measure.

**The ceiling decomposition.** In a generative setting, selection cannot cover
what generation never proposed. We therefore report every coverage number as
achieved = ceiling × efficiency, where the ceiling is the coverage of the
reference by the union of *all* candidates the generator produced, and
efficiency is the fraction of that ceiling the selector captured. A low ceiling
with high efficiency says the axes are too narrow and more search is wasted —
only support expansion (recursive refinement) can help; the reverse says the
selector is the problem. Without the split, "we lost on coverage" gives no
indication of what to change.

## III.2c The comparison that proves the fork

Part II's §II.5 compares RAC-coverage with the released Alpaca, PersonaHub
and WizardLM corpora under a controlled protocol (same 175 human seeds, same
generator, budget matched in generator calls, evaluation on a reference half
nothing ever read). RAC-coverage places first of twelve — 0.4441 against Alpaca's
0.3722, at one-twentieth of Alpaca's budget, with the field's highest
precision. Two rows of its ablation matter to this joint paper beyond the
ranking.

First, **the orthogonalization row**. The configuration applying Part I's
orthogonalized conditioning to the coverage objective scores 0.2941, below
plain conditioning's 0.3147: orthogonalization steers generation away from the
occupied span, which is away from the reference-dense core coverage is paid to
fill. Each part's load-bearing tool, applied to the other's objective, reduces
performance. The two scoring rules fork because they must.

Second, **the Vendi row**. The winning coverage corpus has the *lowest*
mean-centered Vendi of any configuration in its cohort (62.1; density 1.69):
it spends items where the reference measure is, near-duplicates included. A
single "diversity score" would rank the coverage winner last. There is no one
number; there are two objectives.

## III.3 What both objectives share

Both are limited by the same thing, and neither optimizer can fix it. Coverage of
a reachable manifold and packing within one are both bounded by the reachable
manifold's dimension, which is set by how far the prompt can move the generator —
Part I §I.3. Both need the same typicality constraint to be well-posed, since a
novelty-seeking score can otherwise be satisfied by output that has left the
manifold altogether. And both are measured through an embedder whose
geometry can invert the result: we report a cross-corpus coverage comparison in
which our worst corpus by every other measure (19.8% exact duplicates) scores
the **highest** coverage, 50× a published corpus's, because at an ε in the 2nd
percentile of reference distances the metric rewards centrality rather than
spread.

## III.4 Practical guidance

1. **Say which objective you mean.** They are different problems with different
   optimal policies, and the vocabulary ("diverse", "varied") does not distinguish
   them.
2. **Count exact duplicates before computing anything.** A 72% duplicate rate
   makes every distance statistic a statistic about duplication.
3. **Publish the pairwise-similarity distribution** your kernel operates on.
   Mean pairwise cosine was 0.883 in one of our corpora and 0.444 in another with
   the same embedder; an ε or a Vendi score means nothing without it.
4. **Measure at the level of the artifact you ship.** Text-embedding similarity
   explains ~2% of the variance in whether rendered images look alike.
5. **Report the procedure with the number.** Our own framework changed three times
   during this work; every result in this paper is tagged with the procedure that
   produced it.
"""


def main():
    md_maxmin = (SRC_MAXMIN / "PAPER.md").read_text()
    md_cov = (SRC_COV / "PAPER.md").read_text()
    body = (HEAD
            + "\n# Part I — Max-min diversity: no two items alike\n\n"
            + renumber(demote(strip_front(md_maxmin)), "I.")
            + "\n\n---\n\n# Part II — Coverage: reach as much as possible\n\n"
            + renumber(demote(strip_front(md_cov)), "II.")
            + JOINT)
    body = fix_superscripts(body)
    (HERE / "paper.md").write_text(body)
    print(f"paper.md: {len(body.split())} words")

    # the markdown refers to figures/<name>.png, so they must land in a
    # figures/ subdirectory next to the .tex, not beside it
    figdir = HERE / "figures"
    figdir.mkdir(exist_ok=True)
    n_fig = 0
    for src in (SRC_MAXMIN / "figures", SRC_COV / "figures"):
        for p in src.glob("*.png"):
            shutil.copy(p, figdir / p.name)
            n_fig += 1
    print(f"copied {n_fig} figures into figures/")

    pre = HERE / ".preamble.tex"
    pre.write_text(PREAMBLE)
    subprocess.run([
        "pandoc", str(HERE / "paper.md"), "-f", "gfm+tex_math_dollars",
        "-t", "latex", "-s", "--toc", "--toc-depth=2",
        "--pdf-engine=xelatex", "-V", "documentclass=article",
        "-V", "fontsize=10pt", "-V", "mainfont=Palatino", "-V", "monofont=Menlo",
        "-H", str(pre), "-o", str(HERE / "paper.tex")], check=True, cwd=HERE)
    print("paper.tex written")
    for _ in range(2):
        r = subprocess.run(["xelatex", "-interaction=nonstopmode", "paper.tex"],
                           cwd=HERE, capture_output=True, text=True)
    pdf = HERE / "paper.pdf"
    if pdf.exists():
        for ext in (".aux", ".log", ".out", ".toc"):
            (HERE / f"paper{ext}").unlink(missing_ok=True)
        pre.unlink(missing_ok=True)
        print(f"paper.pdf written ({pdf.stat().st_size // 1024} KB)")
    else:
        print("xelatex failed:\n", (r.stdout or "")[-1500:])


if __name__ == "__main__":
    main()
