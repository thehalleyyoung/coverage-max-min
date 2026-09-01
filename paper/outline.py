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
No measure here is limited by its optimizer. What determines whether a corpus can
keep growing without collapsing into paraphrase, and whether it can blanket the
space it is drawn from, is the dimension of the generator's conditional support
relative to the manifold it lives in. Saturation arrives at rate
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

The reversal between the two classes is the finding we would keep if we could
keep only one. Each objective's characteristic tool costs the other measurably:
greedy k-center wins min-gap in both domains and finishes last on coverage, and
orthogonalized conditioning, load-bearing under max-min, scores below doing
nothing at all under coverage. Pointing the loop changes exactly two things, how
an axis is scored and how a candidate is chosen, and those two are enough to
invert which method looks best. A corpus is diverse with respect to an objective,
and the objective has to be named before the number means anything.

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


INTRO = """
Ask a language model for a poem ten times and you get ten poems. Ask it ten
thousand times and you get a few hundred poems and a great deal of paraphrase.
Any fixed conditional distribution behaves this way under repeated sampling,
whatever model supplies it: the distribution has a shape, and sampling traces
that shape ever more densely rather than expanding it.

The practical version of this is now everywhere. Synthetic training data is worth
what it adds to the training set. An evaluation suite that clusters in a few
families gives false assurance. Test-item banks, red-team prompt sets, persona
corpora and augmentation pipelines all consume generated text in bulk, and all of
them degrade in a way that per-item quality checks do not detect: every item is
fine, and the collection is redundant. The failure is a property of the
collection, and the corpora shipped today have it badly. Asked a reasonable
psychometric question ten thousand times, a strong model returns a bank that is
73.6% byte-identical duplicates, and of a 2,500-item sample only 382 items can
legally coexist on one exam form.

This paper presents **Recursive Axis Conditioning** (RAC), a generation loop
that fixes this well enough to beat the released state of the art at a twentieth
of its budget, and reports two results from building it.

The first is that the loop works, and by margins that do not depend on scale.
Against Alpaca, PersonaHub and WizardLM Evol-Instruct on coverage of a held-out
human reference, RAC places first of twelve corpora from 2,400 generator calls
against Alpaca's 52,000. On psychometric item generation it produces a bank with
no enemy pair at all at a judged radius, where the naive bank yields 15.3% of its
nominal capacity and human-written MMLU yields 94.9%.

The second result was not what we set out to find. Corpus objectives fall into
two classes: *covering* the reachable space, and *packing* it so that no two
items collide. These are usually spoken of interchangeably, as two ways of asking
for a diverse corpus. They are not interchangeable, and the machinery that is
load-bearing for one is actively harmful to the other in both directions.
Greedy k-center, the classical packing algorithm, wins the min-gap metric in both
of our domains and finishes last on coverage, at a sixth of random.
Orthogonalized conditioning, the component that makes RAC work under packing,
*reduces* coverage below plain conditioning when carried across. The two
highest-Vendi selectors we measure are the two worst covering ones. Section 4
develops this, and it is the reason the same loop has to be pointed deliberately
rather than tuned once and reused.

Underneath both results is a single limit, which §5 makes precise: no selection
rule can outrun the conditional support of the generator it is drawing from, so
the only move that changes the asymptote is one that widens that support.
"""


MEASURES = """
Every number below is one of the following. We report several because they
disagree, and a corpus that one of them calls healthy another calls collapsed.

**Exact-duplicate rate.** The fraction of the corpus that is byte-identical to
something else in it. Unambiguous, cheap, and the first thing to compute: a
naively prompted psychometric corpus is 73.6% duplicates, and no embedding,
threshold or interpretation is needed to see it. It is also blind to paraphrase.
Deduplicating that corpus removes 2,726 identical copies of one question and
leaves behind the same question with one word changed, the same question with
its options reshuffled, and the same question on different numbers.

**Distinct-*n*, self-repetition, *n*-gram Vendi.** Surface statistics over token
sequences. They catch templating that duplicate rate misses and they read
meaning not at all. A poetry corpus with a 0.000 duplicate rate and a distinct-2
of 0.610 — healthy by both — has eight of eight sampled poems opening *At
dusk/dawn, the windows/river/rooftops gather …*.

**Mean-centered embedding Vendi.** The exponential of the von Neumann entropy of
the corpus Gram matrix, an effective number of distinct items. It summarises the
whole spectrum, which makes it insensitive to local structure: on the poetry
pair above it reads 37.00 for the templated corpus and 38.81 for the varied one,
a difference the page makes in a second. Centering matters as much as the
statistic — the shared mean direction of text embeddings compresses the
uncentered score by roughly 2.7×, so an uncentered Vendi is reporting the cone
as much as the content.

**Median nearest-neighbour distance.** How much room the typical item has. It
registers the poetry difference the Vendi Score misses, 0.089 against 0.239, and
it is the statistic that goes to exactly zero when the median item has a perfect
twin. It says nothing about the shape of the corpus as a whole.

**Worst-case nearest-neighbour distance (the packing radius).** The minimum over
all pairs. It is the only measure here under which a single collision is a
defect regardless of everything else, which is what an exam bank needs: two
items testing the same rule are a security failure whatever the other 2,498
items look like. Optimizing it is the max-min objective of §4.2.

**Coverage, density, precision and recall against a reference.** Ratios computed
against a corpus of real examples, using reference-side k-NN radii so nothing is
tunable per corpus. These are the only measures here that know what the space is
supposed to look like, and the only ones that let corpora of different scales be
compared, which is why the head-to-head against released corpora uses them.
Covered fraction at a *fixed* radius does not survive that comparison: it
correlates −0.991 with within-corpus spacing, so it ranks corpora by how tightly
they cluster rather than by how much they reach.

**A judged threshold.** Where an application defines the failure, the radius can
be measured instead of assumed. Adjudicating 236 item pairs blind puts the
enemy-item radius for exam questions at δ = 0.0354 on this embedder, and that
number, rather than a convention, is what a usable-capacity table rests on.

**Measures on the rendered artifact.** When the text is an instruction to a
second generative model, the corpus that matters is the rendered one. Text
embeddings predict rendered-image similarity at *r* = 0.170, so a text-side
measure explains about 3% of the variance in what the reader receives, and every
text-side method in this literature, ours included, is optimizing a proxy.

Two lessons run through the rest of the paper. The measure has to be named for a
diversity claim to carry information, and it has to be computed at the level of
the artifact being shipped.
"""


MAXMIN_DEF = r"""
Under an unbounded horizon the objective is a two-term score evaluated against
everything generated so far. Let *E* be an embedding oracle and let
*X<sub>n</sub>* = {*x*<sub>1</sub>, …, *x<sub>n</sub>*} be the corpus. For a
candidate *c*,

$$
J_\lambda(c \mid X_n) \;=\; \lambda \cdot \underbrace{\frac{1}{n}\sum_{i=1}^{n}\lVert E(c) - E(x_i)\rVert}_{\text{anchor: keep it typical}} \;-\; (1-\lambda)\cdot \underbrace{\min_{i \le n} \lVert E(c) - E(x_i)\rVert}_{\text{repulsion: keep it new}}
$$

and we accept the candidate minimizing *J*<sub>λ</sub>. The anchor keeps
generation on the manifold, the repulsion pushes it away from what already
exists, and λ trades them off. Both terms are cheap to maintain online: the
anchor decomposes into distance-to-centroid plus a candidate-independent
constant, so ranking by it is *O*(*D*) per candidate and *O*(1) in *n*, and the
repulsion needs one nearest-neighbour query, which we bound by subsampling.

Neither term is where the difficulty lives, as §5 shows. Both
behave predictably; what does not is the set of candidates they are evaluated
on.
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
Covering and packing are routinely treated as two phrasings of one goal. On real
corpora they rank methods almost oppositely, and each one's characteristic tool
costs the other measurably. This section establishes that, then explains why it
happens and what follows for the loop.

Both objectives are stated carefully first, since the reversal is easy to
mistake for a tuning artifact. Coverage is the right question when the
corpus stands for a population, as an evaluation suite or a training set does:
what fraction of the space has an exemplar? Packing is the right question when a
single collision is a defect on its own, as in an exam bank where two items
testing the same rule are a security failure whatever the other items look like.
Both are named below, and then measured against each other.
"""

BRIDGE_THEORY = """
Nothing below depends on which measure was chosen. The results in this section
bound what a fixed prompt can reach at all, and a measure computed on the
resulting corpus inherits that bound whatever it is measuring. The first group
concerns the conditional support and applies to any selection rule; the second
concerns the covering functional in particular, and is what makes greedy
selection defensible when coverage is the measure.
"""

OUTLINE = [
    ("Results", [
        ("text", LEAD),
    ]),
    ("Introduction", [
        ("text", INTRO),
        ("sub", "The oracle asymmetry", "mm", "The oracle asymmetry"),
        ("sub", "Contributions", "text", CONTRIB),
    ]),
    ("Related work", [
        ("mm", "Related work"),
        ("cv", "Related work"),
    ]),
    ("What helps one objective hurts the other", [
        ("text", BRIDGE_REL),
        ("sub", "Max-min", "text", MAXMIN_DEF),
        ("sub", "The reachable manifold and the conditional slice", "mm", "Setup"),
        ("sub", "Coverage", "cv", "Problem statement"),
        ("sub", "Where they coincide", "iii", "Are these dual problems?"),
        ("sub", "Coverage is not packing", "cv", "Coverage is not packing"),
        ("sub", "The reversal, measured", "iii", "The dissociation, measured"),
        ("sub", "Why the scoring rule has to fork", "iii", "Why the scoring rule has to fork"),
        ("sub", "The ablation that shows the fork", "iii", "The comparison that proves the fork"),
        ("sub", "What both classes share", "iii", "What both objectives share"),
        ("sub", "Practical guidance", "iii", "Practical guidance"),
    ]),
    ("Theory: the limit both classes share", [
        ("text", BRIDGE_THEORY),
        ("sub", "Saturation at a fixed prompt", "mm", "Fixed-prompt saturation"),
        ("sub", "The slice deficit", "mm", "The slice deficit"),
        ("sub", "Transversality: which prompt motion helps", "mm", "Transversality: which prompt motion helps"),
        ("sub", "Breadth versus depth", "mm", "Breadth versus depth"),
        ("sub", "What this means for the packing objective", "mm", "What this means for the objective"),
        ("sub", "The coverage functional is monotone submodular", "cv", "The coverage functional is monotone submodular"),
        ("sub", "Greedy and its guarantee", "cv", "Greedy and its guarantee"),
        ("sub", "Sequential generation is streaming greedy", "cv", "Sequential generation is streaming greedy"),
        ("sub", "Estimation error of the Monte-Carlo coverage", "cv", "Estimation error of the Monte-Carlo coverage"),
        ("sub", "Allocation under a known budget", "cv", "Allocation under a known budget"),
        ("sub", "Numerical verification", "cv", "Numerical verification of the theory"),
    ]),
    ("Method: Recursive Axis Conditioning", [
        ("mm", "Method"),
        ("sub", "The stack", "mm", "The stack"),
        ("sub", "Scoring candidate axes", "mm", "The question the axis scoring answers"),
        ("sub", "The four factors", "mm", "The four factors"),
        ("sub", "Choosing values, and recursing", "mm", "Choosing values, and recursing"),
        ("sub", "Eliciting and refining latent behaviors", "mm", "Eliciting and refining latent behaviors"),
        ("sub", "The covering instantiation", "cv", "Method"),
        ("sub", 'Why "orthogonalize", not "randomize"', "mm", 'Why "orthogonalize", not "randomize"'),
    ]),
    ("Experiments", [
        ("mm", "Experiments: 43,000 generations, three domains, seven methods"),
        ("sub", "What we measure, and what each measure misses", "text", MEASURES),
        ("sub", "Domains", "mm", "Domains"),
        ("sub", "Methods compared", "mm", "Methods compared"),
        ("sub", "Duplication under repeated sampling", "mm", "Duplication under repeated sampling"),
        ("sub", "Literal and latent diversity move in opposite directions", "mm", "Literal and latent diversity move in opposite directions"),
        ("sub", "Competitive comparison at matched *n*", "mm", "Competitive comparison at matched *n*"),
        ("sub", "Head-to-head against released instruction corpora", "cv", "Comparison with released instruction corpora"),
        ("sub", "Protocol", "cv", "Protocol"),
        ("sub", "The configuration", "cv", "The configuration"),
        ("sub", "Results", "cv", "Results"),
        ("sub", "What the coverage score buys downstream", "cv", "What the coverage score buys downstream"),
        ("sub", "Notes", "cv", "Notes"),
        ("sub", "A live coverage pilot", "cv", "Live pilot"),
        ("sub", "Setup and what it cost", "cv", "Setup and what it cost"),
        ("sub", "Literal versus latent diversity, and a kernel caveat", "cv", "Literal vs latent diversity, and a kernel caveat"),
        ("sub", "The proxy problem: text diversity is nearly blind to image diversity", "mm", "The proxy problem: text diversity is nearly blind to image diversity"),
        ("sub", "Attractor mining works, and is legible", "mm", "Attractor mining works, and is legible"),
        ("sub", "The exam bank: a judged enemy-item radius", "mm", "The exam bank: a judged enemy-item radius"),
        ("sub", "Head-to-head against a human-written exam bank", "mm", "Head-to-head against a human-written exam bank"),
        ("sub", "Which metrics are independent of the objective", "mm", "Which metrics are independent of the objective"),
        ("sub", "Literal-space repulsion: fixing what the embeddings cannot see", "mm", "Literal-space repulsion: fixing what the embeddings cannot see"),
        ("sub", "Audio: prompts, the steered corpus, and embedder dependence", "mm", "Audio: prompts, the steered corpus, and embedder dependence"),
        ("sub", "Scale and cost", "mm", "Scale and cost"),
    ]),
    ("Limitations", [
        ("text", LIMITATIONS),
    ]),
    ("Conclusion", [
        ("text", CONCLUSION),
    ]),
]
