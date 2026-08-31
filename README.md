# Recursive Axis Conditioning for Diverse Synthetic Data Generation

A synthetic corpus is judged by a measure, and the measure is chosen before the corpus
is. Duplicate rate, distinct-*n*, mean-centered Vendi, nearest-neighbour distance and
coverage against a reference disagree often enough that a corpus one calls healthy
another calls collapsed. This is a method for improving a *chosen* measure, and an
account of how much of it changes when the measure does. The two we take furthest:

- **Coverage** — *in n turns, reach as much of the space as possible.*
- **Max-min** — *in n turns, make no two items resemble each other.*

They pull in different directions. Coverage will happily place two items near
each other if between them they reach a large region; max-min will happily leave
most of the space empty as long as nothing collides. One method, **Recursive Axis
Conditioning** (RAC), serves both, and the objective enters it at only two points:
how a candidate axis is scored, and how one of K candidates is selected. We measure both on the same
generator, the same embedders and the same budget, and we find they disagree
sharply: **greedy k-center wins min-gap and finishes last on coverage** — a sixth
of random — while the highest-Vendi selectors are among the worst covering ones.

The site: **https://thehalleyyoung.github.io/coverage-max-min/**

---

## What's here

```
paper/      the paper: assembler, Markdown, LaTeX source and PDF
index.html  the GitHub Pages site — the paper with figures embedded
code/       every script needed to reproduce the numbers
figures/    generated figures
data/       result JSONs, provenance registry, mined ledgers
```

## The short version

Conditioned on a fixed prompt, a language model's output concentrates on a
submanifold of dimension *m* far below the dimension *d* of the space it could
reach. Fifteen numerical checks confirm the consequences: novelty at fixed prompt
decays as *n*<sup>−1/m</sup> rather than *n*<sup>−1/d</sup>; a single prompt
ε-covers a vanishing ε<sup>d−m</sup> fraction; only prompt motion *transverse* to
the occupied span raises the ceiling; and unconstrained max-min selection is
inconsistent — it picks an off-manifold candidate essentially whenever one appears.

We assume an embedding oracle and **no inverse**. We can compute exactly where the
next item should land and have no way to decode that point into text. Everything
follows from that: the system must propose, measure and select rather than solve,
and the only steering handles are language-valued.

So we elicit the latent axes from the generator itself and rank them with a
**calculus of diversity** — spread × transversality × independence × headroom —
which has two forms, one per objective, differing on exactly the two terms you
would expect (see `code/calculus.py`).

## Headline measurements

All from real generations, ~$40 of API spend. Numbers are labelled by the
procedure that produced them (`data/provenance.json`); the framework changed
during the work and earlier results are kept and annotated rather than restated.

| finding | number |
|---|---|
| psychometric naive prompting, exact-duplicate rate | **72.3%** (one item ×1,637 in 5,878) |
| same, at temperature 1.6 | 61.6% |
| same, with latent conditioning | 0.2% |
| text-embedding vs rendered-image pairwise similarity | ***r* = 0.155** |
| ours vs best published method (centered Vendi, DALL·E) | **+36%** |
| ours vs human-written MMLU | wins duplication, self-repetition, *n*-gram Vendi; 69% of its semantic diversity |
| coverage-greedy vs best of 5 baselines | 1.043× / 1.091× (leakage-free) |
| k-center: min-gap rank / coverage rank | **1st / last** |

## Reproducing

```bash
pip install numpy scipy matplotlib transformers torch soundfile pillow
python3 code/verify_theory.py     # 8/8 theorem checks
python3 code/verify_slices.py     # 7/7 conditional-dimension checks
python3 code/calculus.py          # calculus self-test, both objectives
```

Live runs need `OPENROUTER_API_KEY`, a local Ollama with `nomic-embed-text`, and
optionally `OPENAI_API_KEY` (images) and a Gemini key (Lyria audio). See
`code/README.md`.

## Licence

Code MIT. Paper CC BY 4.0. Generated corpora are model outputs and are released
alongside the code for replication.
