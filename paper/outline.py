"""The unified outline: one paper, each topic covered once for both objectives.

Every entry is (source, key) drawn from the two working papers, or ("text", md)
for connective material written here. Numbers are assigned by position, and the
old->new map is derived from the same table, so cross-references cannot drift.
"""


LIMITATIONS = """
- **One embedding oracle.** All geometry is `nomic-embed-text` geometry with
  cosine distance, on both objectives. Whether "diverse" or "covered" under one
  embedder transfers to another is untested here, and it is a real threat to any
  embedding-based claim, ours included. Section 7.13 measures a correlation of
  only *r* = 0.170 between text-embedding and CLIP-image similarity on rendered
  outputs, so for any domain with a downstream rendering step the objective
  should be defined in the space the artifact actually occupies. We did not do
  that.
- **ε is chosen, not learned.** Every covering guarantee is stated per-ε, and our
  calibration from quantiles of reachable nearest-neighbour distance is a
  heuristic. Section 5.11 shows what optimizing at the wrong radius costs. A
  multi-resolution objective, integrating coverage over a prior on ε, is the
  obvious next step.
- **The pool is the measure.** Coverage is relative to the reachable pool
  throughout. Behaviors the base generator cannot emit are invisible until
  refinement opens them, and the pool refresh is only as good as the refinement
  trigger that fires it.
- **Matched-*n* does not neutralize generator quality.** In the instruction
  head-to-head the released corpora are 20–70× larger than ours and were written
  by different, mostly stronger generators, with human curation in at least one
  case. Matched-*n* sampling controls for size and cannot control for the model
  that wrote the items; PersonaHub's persona catalogue alone is orders of
  magnitude larger than any axis lattice we elicit.
- **Selection bias on the estimation pool.** The head-to-head is scored on a
  reference half no pipeline ever reads, but the live pilot's selection and
  reporting share one 240-item pool. The direction of the bias is stated where
  it appears; a larger live run should split them.
- **Judge bias is unmeasured.** Language-model judges prefer fluent, typical
  text, which is the bias that would work against a diversity system. We use a
  judge for craft and validity and did not calibrate its error against human
  raters. The exam-item enemy radius is likewise adjudicated by a model under a
  psychometric rubric rather than by a credentialed psychometrician, and δ is
  specific to this embedder and this item type.
- **Additive spec composition.** The axis scoring treats conditioning attributes
  as composing additively in embedding space, which is what makes a spec's
  position predictable before it is generated. Real interactions between prompt
  attributes are not additive.
- **Quality is scalar.** Craft is not one number, and collapsing it to one lets a
  system trade away dimensions of quality the judge does not score.
"""

CONCLUSION = """
Neither objective is limited by its optimizer. What determines whether a corpus
can keep growing without collapsing into paraphrase, and whether it can blanket
the space it is drawn from, is the dimension of the generator's conditional
support relative to the manifold it lives in. Saturation arrives at rate
*n*<sup>−1/*m*</sup> rather than *n*<sup>−1/*d*</sup>, a single prompt covers an
ε<sup>*d*−*m*</sup> fraction of what the model could write, and only transverse
prompt motion raises the ceiling. Everything else buys a constant factor against
a problem that is asymptotic.

Given an embedding oracle and no inverse, the system cannot compute its way to
the next item; it can only choose what to condition on. Recursive Axis
Conditioning is our answer to that choice, and its most valuable output is the
refine signal: the moment the scoring reports that nothing available is
transverse any more is the moment the horizon has been reached, and the only
remaining move is to ask the generator to subdivide its own vocabulary of
variation.

The two objectives share that machinery and diverge exactly twice, at how an
axis is scored and at how a candidate is chosen, and those two differences are
enough to reverse the ranking of methods. Greedy k-center wins min-gap in both
domains and finishes last on coverage; orthogonalized conditioning, which is
load-bearing under max-min, scores below plain conditioning under coverage. A
corpus is not diverse or undiverse in the abstract. It is diverse with respect
to an objective, and the objective has to be named before the number means
anything.

The measurements make the stakes concrete in a way the theory could not. A
strong model asked a reasonable question ten thousand times returns the same
item 2,726 times; the temperature knob barely moves that and conditioning nearly
erases it. Two diversity metrics computed on one growing corpus point in
opposite directions. The text embeddings every method in this literature
optimizes explain about three percent of the variance in whether the rendered
images look alike. And a method that grows the space it is covering has to say
so when it reports how much of that space it covered. Each of those is a reason
to distrust a single diversity number, and together they are the argument for
the practice we ended up recommending: measure at the level of the artifact you
are shipping, report the literal and the latent separately, name the
denominator, and count your duplicates before computing anything else.
"""

# ---------------------------------------------------------------- section 1
LEAD = """
Synthetic corpora are judged by a measure, and the measure is chosen before the
corpus is. This paper is about building a generator loop that improves such a
measure. The method is **Recursive Axis Conditioning** (RAC), and the two
measures we take it furthest on are **coverage**, which asks the corpus to reach
as much of the space as possible in *n* generator calls, and **max-min**, which
asks that no two of the *n* items resemble each other. What the method achieves
on each:

**Coverage of a held-out human-written reference.** Twelve corpora, matched
evaluated *n* = 450, scored by the Naeem et al. (2020) estimator with
reference-side k-NN radii, reported as AUC over k ∈ {3, 5, 10, 20}. The radii
are a property of the reference alone, so nothing is tunable per corpus, and no
corpus in the table was aimed at the evaluation half.

| rank | corpus | coverage AUC | precision |
|---|---|---|---|
| 1 | **RAC-coverage, retrieval-aimed (2.4k)** | **0.4441** | **0.973** |
| 2 | Alpaca (Self-Instruct, 52k) | 0.3722 | 0.87 |
| 3 | RAC-coverage, selective 1-of-8 (304) | 0.2591 | 0.93 |
| 4 | PersonaHub (50k) | 0.2532 | 0.78 |
| 5 | WizardLM Evol-Instruct (143k) | 0.2329 | 0.77 |
| 6 | Self-Instruct (reimplemented, same seeds and budget) | 0.2188 | |
| 7–9 | budget-matched ablation arms | 0.2102–0.2127 | |
| 10–12 | unseeded arms (450 each) | 0.0947–0.1008 | |

First place, ahead of Alpaca by 19%, on one twentieth of Alpaca's generation
budget and with the highest precision in the field. PersonaHub and WizardLM
finish below our 304-item selective arm despite 20–60× the scale.

**Max-min at matched *n*.** Against five published methods on two text domains,
RAC wins every literal and latent measure. On psychometric items at *n* = 1,058:

| arm | exact-dup ↓ | distinct-2 ↑ | self-repetition ↓ | *n*-gram Vendi ↑ | centered Vendi ↑ | median NN dist ↑ |
|---|---|---|---|---|---|---|
| naive | 0.668 | 0.114 | 0.855 | 11.8 | 7.1 | 0.000 |
| high temperature | 0.660 | 0.119 | 0.845 | 12.2 | 7.0 | 0.000 |
| Self-Instruct | 0.026 | 0.193 | 0.485 | 288.8 | 59.3 | 0.033 |
| Evol-Instruct | 0.025 | 0.224 | 0.470 | 307.4 | 75.0 | 0.047 |
| persona conditioning | 0.004 | 0.377 | 0.184 | 545.6 | 99.7 | 0.130 |
| **RAC** | **0.000** | **0.429** | **0.053** | **644.3** | **124.8** | **0.210** |

Against human-written MMLU at matched *n* = 1,000, RAC wins exact duplication,
4-gram self-repetition and *n*-gram Vendi, and reaches 69% of the human bank's
mean-centered semantic diversity.

The rest of the paper says what the method is, why the two objectives need
different scoring rules, and what limits both of them.
"""

CONTRIB = """
- **A generator loop that improves a chosen measure** (§6). Language-valued axes
  elicited from the generator, ranked by a scoring rule, with the exhausted ones
  split recursively into conditional sub-axes. The measure enters at two points
  only: how an axis is scored, and how one of K candidates is selected, so
  pointing the loop at a different measure means changing those two and nothing
  else.
- **A theory of what limits both** (§5). Conditioned on a fixed prompt, the
  output concentrates on a submanifold of dimension *m* far below the dimension
  *d* of the reachable space, and the consequences are the same for packing and
  for covering.
- **Evidence that the measure has to be named** (§4). Greedy k-center wins min-gap
  in both domains and finishes last on coverage, and each measure's characteristic
  tool damages the other's score, so "diverse" without a named measure carries no
  information.
- **Head-to-head against released corpora** (§7), on a scale-free estimator
  against a reference no corpus was aimed at, and against a human-written exam
  bank under a judged enemy-item radius.
"""

BRIDGE_REL = """
Everything so far has been about improving *a* measure. Which one is not a detail
that can be left until evaluation, because the measure changes the method. This
section states the two we study, shows where they agree, and shows where a loop
tuned for one is actively worse at the other.

The two are chosen because they sit at opposite ends of what practitioners
actually ask for. Coverage is the right question when the corpus is an evaluation
suite or a training set meant to represent a population: what fraction of the
space has an exemplar? Max-min is the right question when any single collision is
a defect, as in an exam bank where two items testing the same rule are a security
failure whatever the rest of the bank looks like. Other measures sit between them
— duplicate rate, mean-centered Vendi, precision against a reference, the
worst-case nearest-neighbour distance — and the machinery below applies to those
too, since what changes from one to the next is the scoring rule and the
selection rule, not the loop that carries them.
"""

BRIDGE_THEORY = """
Everything below constrains both objectives. The first group concerns what a
fixed prompt can reach at all, and applies whatever is done with the candidates
it produces; the second concerns the covering functional in particular.
"""

OUTLINE = [
    ("Results", [
        ("text", LEAD),
    ]),
    ("Introduction", [
        ("mm", "1"),
        ("sub", "The oracle asymmetry", "mm", "1.1"),
        ("sub", "Contributions", "text", CONTRIB),
    ]),
    ("Related work", [
        ("mm", "2"),
        ("cv", "8"),
    ]),
    ("Which measure, and what changes with it", [
        ("text", BRIDGE_REL),
        ("sub", "Max-min, stated", "mm", "3.1"),
        ("sub", "Coverage, stated", "cv", "2"),
        ("sub", "Where they coincide", "iii", "2b"),
        ("sub", "Coverage is not packing", "cv", "3.4"),
        ("sub", "Where they part, measured", "iii", "1"),
        ("sub", "Why the scoring rule has to fork", "iii", "2"),
        ("sub", "The comparison that shows the fork", "iii", "2c"),
        ("sub", "What both objectives share", "iii", "3"),
        ("sub", "Practical guidance", "iii", "4"),
    ]),
    ("Theory: what limits both", [
        ("text", BRIDGE_THEORY),
        ("sub", "Saturation at a fixed prompt", "mm", "3.2"),
        ("sub", "The slice deficit", "mm", "3.3"),
        ("sub", "Transversality: which prompt motion helps", "mm", "3.4"),
        ("sub", "Breadth versus depth", "mm", "3.5"),
        ("sub", "What this means for the packing objective", "mm", "3.6"),
        ("sub", "The coverage functional is monotone submodular", "cv", "3.1"),
        ("sub", "Greedy and its guarantee", "cv", "3.2"),
        ("sub", "Sequential generation is streaming greedy", "cv", "3.3"),
        ("sub", "Estimation error of the Monte-Carlo coverage", "cv", "3.5"),
        ("sub", "Allocation under a known budget", "cv", "3.6"),
        ("sub", "Numerical verification", "cv", "7"),
    ]),
    ("Method: Recursive Axis Conditioning", [
        ("mm", "5"),
        ("sub", "The stack", "mm", "5.1"),
        ("sub", "Scoring candidate axes", "mm", "4.1"),
        ("sub", "The four factors", "mm", "4.2"),
        ("sub", "Choosing values, and recursing", "mm", "4.3"),
        ("sub", "Eliciting and refining latent behaviors", "mm", "5.2"),
        ("sub", "The covering instantiation", "cv", "4"),
        ("sub", 'Why "orthogonalize", not "randomize"', "mm", "5.3"),
    ]),
    ("Experiments", [
        ("mm", "6"),
        ("sub", "Domains", "mm", "6.1"),
        ("sub", "Methods compared", "mm", "6.2"),
        ("sub", "Duplication under repeated sampling", "mm", "6.3"),
        ("sub", "Literal and latent diversity move in opposite directions", "mm", "6.4"),
        ("sub", "Competitive comparison at matched *n*", "mm", "6.5"),
        ("sub", "Head-to-head against released instruction corpora", "cv", "5"),
        ("sub", "Protocol", "cv", "5.1"),
        ("sub", "The configuration", "cv", "5.2"),
        ("sub", "Results", "cv", "5.3"),
        ("sub", "Notes", "cv", "5.4"),
        ("sub", "A live coverage pilot", "cv", "6"),
        ("sub", "Setup and what it cost", "cv", "6.1"),
        ("sub", "Literal versus latent diversity, and a kernel caveat", "cv", "6.2"),
        ("sub", "The proxy problem: text diversity is nearly blind to image diversity", "mm", "6.6"),
        ("sub", "Attractor mining works, and is legible", "mm", "6.7"),
        ("sub", "The exam bank: a judged enemy-item radius", "mm", "6.8"),
        ("sub", "Head-to-head against a human-written exam bank", "mm", "6.9"),
        ("sub", "Which metrics are independent of the objective", "mm", "6.10"),
        ("sub", "Literal-space repulsion: fixing what the embeddings cannot see", "mm", "6.11"),
        ("sub", "Audio: prompts, the steered corpus, and embedder dependence", "mm", "6.12"),
        ("sub", "Scale and cost", "mm", "6.13"),
    ]),
    ("Limitations", [
        ("text", LIMITATIONS),
    ]),
    ("Conclusion", [
        ("text", CONCLUSION),
    ]),
]
