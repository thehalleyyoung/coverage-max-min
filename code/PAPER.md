# Infinite-Horizon Diversity

### Why a fixed prompt has a finite novelty budget, and what to spend instead

---

## Abstract

We study the problem of generating a corpus that stays diverse as it grows without bound. The objective is a convex combination of two terms evaluated against everything generated so far: the mean distance from a new item's embedding to the existing corpus, which we want small because it keeps items typical and on-manifold, and the minimum distance to the existing corpus, which we want large because it keeps items non-redundant. We call the resulting problem *infinite-horizon diversity*, and we are interested specifically in its asymptotics: what happens to the objective, and to the cost of satisfying it, as *n* grows.

Our central claim locates the interesting structure in the **support** rather than in the objective. Conditioned on a fixed prompt *x*, a language model's output distribution concentrates on a submanifold of dimension *m* that is far smaller than the dimension *d* of the reachable semantic manifold. We prove and numerically verify four consequences of that gap. Novelty at fixed prompt decays like *n*<sup>−1/*m*</sup>, not *n*<sup>−1/*d*</sup>, so saturation arrives much sooner than the ambient dimension suggests (measured slopes −0.472, −0.309, −0.195 for *m* = 2, 3, 5 against predictions −0.5, −0.333, −0.2). A single prompt ε-covers a vanishing ε<sup>*d*−*m*</sup> fraction of the manifold. Only prompt motion *transverse* to the already-occupied span raises the reachable dimension: parallel motion leaves the union's effective dimension at 1.9, transverse motion takes it to 11.2. Under free prompt-switching the optimal number of samples per prompt is exactly one; an interior optimum appears only once switching has overhead (*n*\* = 10 at cost *c* = 3, *n*\* = 30 at *c* = 30).

We assume an embedding oracle mapping text to a fixed-size vector, and — critically — **no reverse oracle**. We can compute exactly where we would like the next item to land and still have no way to decode that point into text. Every architectural choice follows from that asymmetry: the system must propose, measure, and select rather than solve, and the only steering handles are language-valued.

We therefore build the generation stack (**Recursive Axis Conditioning**, RAC) around latent behaviors elicited from the generator itself, organized by an **axis-scoring rule** that ranks candidate latent variables by the product of four measurable factors (spread, transversality, independence, and headroom), and selects each variable's most-different values by farthest-point search in level space. When the score reports that no available variable clears a promise floor, it emits a *refine* decision, and the generator is asked to split the exhausted variable into finer sub-variables. The recursion is the only mechanism in the paper that changes the asymptote rather than the constant.

We test this on **43,000 real generations** from `openai/gpt-5.6-luna` across three domains — DALL·E instructions for post-modern artworks, psychometric exam items, and instrumental-music prompts — measuring diversity at three levels: literal (*n*-gram), latent (text embedding), and, where the artifact is not text, in the space the product actually occupies (CLIP for rendered images, CLAP and MERT for rendered audio). Four results stand out. Asked the same reasonable question ten thousand times, the model returns a psychometric corpus that is **72.3% exact duplicates**, one item repeated 1,637 times; raising the temperature to 1.6 leaves it at 61.6%, while latent conditioning drops it to 0.2%. Literal and latent diversity move in **opposite directions** as the corpus grows. Text-embedding similarity predicts *rendered-image* similarity at only **r = 0.170**, so every text-side diversity method (ours included) optimizes a proxy explaining about 3% of the variance in what the user receives. And against five real published methods (Self-Instruct, Evol-Instruct, Persona-Hub, high-temperature, naive) our method wins every metric, beating the strongest by 36% in mean-centered Vendi; against **human-written MMLU** it wins on lexical diversity and self-repetition while reaching 69% of a human exam bank's semantic diversity.

---

## 1. Introduction

Ask a language model for a poem ten times and you get ten poems. Ask it ten thousand times and you do not get ten thousand poems; you get a few hundred poems and a great deal of paraphrase. Any fixed conditional distribution behaves this way under repeated sampling, whatever model supplies it. The distribution has a shape, and repeated sampling traces that shape ever more densely rather than expanding it.

The practical version of this problem is now everywhere. Synthetic training data is valuable in proportion to how much it is *not* already in the training set. Evaluation suites that cluster in a few families give false assurance. Test-item banks, red-team prompt sets, persona corpora, and augmentation pipelines all consume generated text in bulk, and all of them degrade in a way that standard quality metrics do not detect: every individual item is fine, and the collection is redundant.

The natural formalization is a two-term objective. Let *E* be an embedding oracle and let *X<sub>n</sub>* = {*x*<sub>1</sub>, …, *x<sub>n</sub>*} be the corpus so far. For a candidate *c*,

$$
J_\lambda(c \mid X_n) \;=\; \lambda \cdot \underbrace{\frac{1}{n}\sum_{i=1}^{n}\lVert E(c) - E(x_i)\rVert}_{\text{anchor: keep it typical}} \;-\; (1-\lambda)\cdot \underbrace{\min_{i \le n} \lVert E(c) - E(x_i)\rVert}_{\text{repulsion: keep it new}}
$$

and we accept the candidate minimizing *J*<sub>λ</sub>. The anchor term keeps generation on the manifold; the repulsion term pushes it away from what already exists. The interesting question is what happens to this as *n* → ∞.

Our answer is that the objective is not where the difficulty lives. Both terms are cheap to maintain online and both behave predictably. The difficulty is that **the set of things the generator can propose is much smaller than it appears**, and the objective is powerless to enlarge it. Selection can only choose among what is offered. If the offering is confined to a thin slice, no selection rule (however clever, however well-tuned its λ) can produce diversity that the proposal distribution does not contain. Worse, as we show, aggressive selection under a fixed proposal distribution systematically degrades quality, because the only way to keep winning a novelty contest inside an exhausted region is to leave the manifold.

### 1.1 The oracle asymmetry

We assume throughout:

> **Assumption (embedding oracle, no inverse).** We have access to *E*: Text → ℝ<sup>*D*</sup>, computable on demand. We have **no** *E*<sup>−1</sup>: ℝ<sup>*D*</sup> → Text.

This is the structural fact that shapes the entire design, and it is worth dwelling on because it is easy to forget when writing down objectives. Given *X<sub>n</sub>* we can compute, exactly and cheaply, the point in ℝ<sup>*D*</sup> that would most improve any of our diversity measures — the direction of least occupied spectral energy, the centre of the largest empty ball, the point maximizing marginal coverage. Knowing that point is worth nothing on its own. There is no procedure that turns a target embedding back into a poem.

Consequently the system cannot *solve* for its next item. It can only:

1. **propose** (sample from *p*(· | *x*) for some prompt *x* we are able to write;
2. **measure**) embed the proposals and score them;
3. **select** — keep one.

All of the control we have is therefore in step 1, in the choice of *x*, and that choice is expressible only in language. This is why the latent variables in our system are *language-valued* (named axes with named levels), rather than continuous codes: they are the only handles that reach the generator. It is also why the axis scoring of §4 exists. **The axis scoring is a surrogate for the missing inverse.** Since we cannot decode the direction we want to travel, we instead score the language-valued conditions we *can* write by how nearly their induced output distributions point that way.

### 1.2 Contributions

- **A conditional-dimension theory of generative saturation** (§3). Four theorems relating the conditional support dimension *m* to the reachable manifold dimension *d*, each verified numerically. The practically important one is that saturation is governed by *m*, which is small, and not by *d*.
- **An axis-scoring rule** (§4). A decision procedure for which latent variable to condition on next and which of its values to use, together with a principled termination/recursion signal.
- **A budget-matched evaluation protocol** (§6). Every method receives the same generator, the same seed material, and the same number of generator calls. This is the fair frame for selection-based diversity methods, which buy diversity by discarding.
- **Three domains with different diversity semantics** (§6): instructions for image generation and prompts for instrumental music, where diversity is a soft objective, and psychometric exam items, where two items closer than δ are a test-security failure and the floor is hard.
- **43,000 real generations** on `openai/gpt-5.6-luna` with locally-computed embeddings, across three domains and seven methods (§6).

---

## 2. Related work

**Diversity metrics.** The Vendi Score (Friedman & Dieng, 2022) is the exponential of the von Neumann entropy of a normalized similarity matrix, interpretable as an effective number of distinct items. We use it throughout, with a computational note: under a linear kernel on unit-normalized embeddings, the *n* × *n* Gram matrix and the *D* × *D* second-moment matrix share their nonzero spectrum, so the score costs *O*(*nD*² + *D*³), rather than *O*(*n*³). We verify this to machine precision (max eigenvalue mismatch 1.1 × 10<sup>−16</sup>). This matters for our setting specifically: an infinite-horizon method needs an evaluation whose cost does not grow superlinearly in the thing it is evaluating.

**Selection and subset choice.** Determinantal point processes model repulsion via volume in feature space. Farthest-point / *k*-center greedy is the classical max–min packing heuristic and we use it as a baseline (`fps_post`) and, at the level of axis values, as a component (§4.3). SemDeDup-style near-duplicate removal at a fixed radius appears as `dedup_post`. All of these are *post hoc* selectors over a fixed pool, which is exactly their limitation in our setting: they cannot change what is in the pool.

**Quality-diversity.** MAP-Elites and its descendants maintain an archive indexed by hand-designed behavior descriptors, seeking an elite per cell. Our axis lattice is a close relative, with two differences that matter here: the descriptors are *elicited from the generator rather than designed by us*, and the lattice *refines itself* when a cell saturates, so the archive's resolution is not fixed in advance.

**Mode collapse in instruction-tuned models.** Repeated sampling from aligned models is well documented to produce low lexical and semantic entropy relative to base models. We take this as our empirical starting point and model it explicitly as skewed mixture weights over style modes.

**Novelty search.** Novelty search in evolutionary computation rewards behavioral distance from an archive. Our repulsion term is a direct analogue, operating over an archive of embeddings rather than of behavior descriptors.

**Coverage.** Part II treats the covering objective (maximizing the volume of a union of ε-balls at a *finite* budget), which is submodular where max–min is not. §7 relates the two.

---

## 3. Theory: conditional dimension governs saturation

### 3.1 Setup

Let *M* ⊂ ℝ<sup>*D*</sup> be the reachable semantic manifold, dim *M* = *d*: the set of embeddings of texts the generator could produce under *some* prompt. For a fixed prompt *x*, let *S<sub>x</sub>* ⊆ *M* be the support of *E*<sub>#</sub>*p*(· | *x*), with dim *S<sub>x</sub>* = *m*.

The empirical claim behind everything below is that **_m_ ≪ _d_**. A prompt fixes topic, stance, form, register, and rhetorical strategy; what remains free is a low-dimensional residue — some lexical choice, some ordering, some surface variation. Our live pilot supports this directly: 60 poems generated from an identical naive prompt have a median nearest-neighbour cosine similarity of 0.911 and a Vendi Score of 2.55, meaning sixty samples behave like roughly two and a half distinct items.

Throughout, *B*(*x*, *r*) is the ball of radius *r* and we write *g<sub>n</sub>* for the min-gap of a fresh draw against a corpus of size *n*.

### 3.2 Fixed-prompt saturation

> **Theorem 1.** Let *X<sub>n</sub>* be *n* i.i.d. draws from a distribution with density bounded above and below on an *m*-dimensional manifold *S*. For a fresh draw *c* from the same distribution, 𝔼[*g<sub>n</sub>*] = Θ(*n*<sup>−1/*m*</sup>).

*Proof sketch.* ℙ(*g<sub>n</sub>* > *r*) = ℙ(no sample in *B*(*c*, *r*)) = (1 − μ(*B*(*c*, *r*)))<sup>*n*</sup>, and μ(*B*(*c*, *r*)) = Θ(*r*<sup>*m*</sup>) by density bounds on an *m*-manifold. So ℙ(*g<sub>n</sub>* > *r*) ≈ exp(−*n*Θ(*r<sup>m</sup>*)), which transitions at *r* = Θ(*n*<sup>−1/*m*</sup>); integrating the tail gives the expectation. ∎

The content is the exponent. With *d* = 32, a naive reading predicts *n*<sup>−1/32</sup> — essentially flat, novelty never exhausted. The true rate at *m* = 3 is *n*<sup>−1/3</sup>, which is roughly a 10× loss of headroom per 1000× of corpus. Measured slopes: −0.472 (*m* = 2), −0.309 (*m* = 3), −0.195 (*m* = 5), against −0.5, −0.333, −0.2. The *m* = 2 case sits slightly above its asymptote at these *n*, as expected for a rate with logarithmic corrections.

> **Corollary 1.1 (oversampling is not a strategy).** Drawing *K* candidates and keeping the most novel multiplies the expected gap by Θ(*K*<sup>1/*m*</sup>) at best. To recover the headroom lost by a 1000× larger corpus, *K* must grow by 1000.

Verified at *m* = 3: best-of-8 yields a gain of 2.19 against *K*<sup>1/3</sup> = 2.00. Selection is a constant-factor fix to an asymptotic problem.

### 3.3 The slice deficit

> **Theorem 2.** If *m* < *d*, then vol<sub>*d*</sub>(*S<sub>x</sub>*) = 0, and the fraction of *M* within ε of *S<sub>x</sub>* is Θ(ε<sup>*d*−*m*</sup>).

*Proof sketch.* The ε-neighbourhood of an *m*-dimensional set in *d* dimensions is a tube of volume Θ(ε<sup>*d*−*m*</sup>) · vol<sub>*m*</sub>(*S<sub>x</sub>*) for ε below the reach of *S<sub>x</sub>*. Dividing by vol<sub>*d*</sub>(*M*) gives the claim. ∎

Measured at *d* = 8, *m* = 2: fitted exponent 5.06 against a predicted 6, with covered fractions falling 0.051 → 0.0001 as ε goes 0.5 → 0.12. The fitted exponent sits below prediction because at the larger ε the tube is no longer thin relative to the manifold, which flattens the low-ε end of the fit; the qualitative claim (coverage collapsing polynomially in ε with a large exponent) is unambiguous.

The practical reading is severe. **A single prompt, sampled infinitely often, covers essentially none of what the model could write.** Not "less than we would like" — a fraction that goes to zero polynomially as the resolution of interest sharpens.

### 3.4 Transversality: which prompt motion helps

Let a prompt schedule induce slices *S*<sub>1</sub>, …, *S<sub>P</sub>*. Write *V*<sub>occ</sub> for the span of the corpus's dominant directions.

> **Theorem 3.** dim(⋃<sub>*j*</sub> *S<sub>j</sub>*) = min(*d*, *m* + rank{*c<sub>j</sub>* − *c*<sub>1</sub>}), where *c<sub>j</sub>* is the centre of *S<sub>j</sub>*. In particular, displacements lying inside span(*S*<sub>1</sub>) contribute nothing to the union's dimension.

*Proof sketch.* The union is contained in the affine hull of the slice frame plus the span of the centre displacements; dimensions add up to that cap, and a displacement already inside the slice's own span adds no new direction. ∎

Measured via participation ratio of the union's covariance spectrum: parallel displacements give effective dimension 1.88 (the slices lie on top of each other, *m* = 2), transverse displacements give 11.18.

![Figure 1. Measured saturation. (a) Expected min-gap of a fresh draw decays as n^(-1/k), with fitted slopes matching theory for k = 3, 8, 32. (b) Best-of-K oversampling buys only K^(1/k).](figures/fig5_scaling.png)

*Figure 1. Measured saturation. (a) Expected min-gap of a fresh draw decays as n^(-1/k), with fitted slopes matching theory for k = 3, 8, 32. (b) Best-of-K oversampling buys only K^(1/k).*


This is the theorem that makes the axis scoring of §4 possible. It measures the value of conditioning on a new latent variable as **how much of its variation lies outside what the corpus already spans** — a computable quantity, in place of the intuition of how different it sounds.

### 3.5 Breadth versus depth

Given budget *B*, split as *P* prompts × *n* samples each, with per-prompt switching cost *c* (eliciting a spec, embedding it, occasionally refining), so *B* = *P*(*c* + *n*).

> **Theorem 4.** With *c* = 0 the coverage-optimal depth is *n*\* = 1. For *c* > 0 the optimum is interior and increases with *c*.

Measured: with free switching, depth 1 achieves coverage 0.995 while spending the entire budget on one prompt achieves 0.091 — a 10.9× difference. With *c* = 3, *n*\* = 10; with *c* = 30, *n*\* = 30.

![Figure 2. Breadth versus depth at fixed budget. With free prompt-switching the optimum is one sample per prompt; an interior optimum appears only once switching is priced.](figures/fig8_breadth_depth.png)

*Figure 2. Breadth versus depth at fixed budget. With free prompt-switching the optimum is one sample per prompt; an interior optimum appears only once switching is priced.*


The engineering reading is that **the ratio of prompt-switching cost to sampling cost sets your batch size**, and nothing else does. If constructing a new spec is as cheap as a generation, generate one item per spec.

### 3.6 What this means for the objective

Returning to *J*<sub>λ</sub>: the anchor term is exactly computable from two running statistics. Writing μ*<sub>n</sub>* for the corpus centroid,

$$
\frac{1}{n}\sum_i \lVert c - x_i\rVert^2 = \lVert c - \mu_n\rVert^2 + \frac{1}{n}\sum_i \lVert x_i - \mu_n\rVert^2
$$

verified to zero relative error. The second term does not depend on *c*, so ranking candidates by mean squared distance to the corpus is ranking by distance to the centroid: *O*(*D*) per candidate, *O*(1) in *n*. The repulsion term needs a nearest-neighbour query, which we bound by subsampling. Neither term is the bottleneck. **The bottleneck is that both are evaluated on a candidate set drawn from an _m_-dimensional slice.**

---

## 4. Scoring candidate axes

### 4.1 The question the axis scoring answers

Theorem 3 says progress requires moving the prompt transversely to what the corpus already spans. Theorem 4 says move often. Together they pose a concrete question at every step: **among the latent variables we could condition on next, which one moves the slice most transversely, per unit of budget?** Since we have no inverse oracle, this is the closest available substitute for computing the ideal next item.

Each candidate axis *a* has language-valued levels *L*(*a*). We embed the level descriptions and read off four quantities.

### 4.2 The four factors

**Spread** — mean pairwise distance between *a*'s level embeddings. Are this axis's values actually different *from each other*? An axis whose levels are near-synonyms cannot separate outputs however it is sampled. This catches the most common failure of LLM-proposed axes: plausible-sounding dimensions whose levels all mean the same thing.

**Transversality** — the fraction of *a*'s level-to-level variation lying outside the corpus's occupied eigenspace (top eigenvectors of the second-moment matrix carrying 80% of spectral energy). This is Theorem 3 made into a number. An axis whose levels differ only along directions the corpus already spans adds density, not reach.

**Independence** — 1 − max principal-angle cosine between *a*'s variation subspace and those of the axes **already in use**. An axis that restates an active axis is redundant however transverse it looks alone. Independence is measured against the active set and never against rival candidates: two equally good candidates would otherwise drive each other's score to zero and rank below a useless axis with no near-twin. Redundancy *among* candidates is a selection problem, handled by greedy selection that re-scores after every pick, so a duplicate is penalized at the point where the redundancy comes into existence.

**Headroom** — a normalized entropy deficit of how unevenly the corpus has sampled *a*'s levels, plus the fraction of levels never used. This is the only time-varying factor; it is what makes the ranking change as generation proceeds rather than being a fixed property of the axis set. An axis can be excellent and already spent.

The **promise** of an axis is the product:

$$
\mathrm{promise}(a) = \mathrm{spread}(a)\cdot\mathrm{transversality}(a)\cdot\mathrm{independence}(a)\cdot\mathrm{headroom}(a)
$$

Multiplicative, not additive, and deliberately so: an axis needs all four, and any one at zero should zero the score rather than be averaged away by the others. A high-spread axis with zero transversality is a distinction without a difference; a perfect axis with zero headroom is already spent.

### 4.3 Choosing values, and recursing

Once an axis is chosen we do not sample its levels uniformly. We take the **max–min subset** of its level embeddings by greedy farthest-point search, seeded deterministically at the level furthest from the level centroid — the packing problem again, one level down.

When the best available promise falls below a floor, the axis scoring emits `decision: "refine"` instead of `"condition"`. This is the **exhaustion signal**, and it is the honest one: it says the current lattice has nothing transverse left to offer, and no amount of further sampling will change that. The system then asks the generator to split the exhausted level into finer sub-levels (§5.2), producing a child axis scored by this same axis scoring. On the ground-truth test with only in-span axes available, the axis scoring correctly returns `refine`.

Cost is *O*(|*A*| · *D*²) per decision and does not grow with *n*.

---

## 5. Method

We call the method **Recursive Axis Conditioning** (RAC): the generator is asked for the language-valued axes along which its own outputs can differ, those axes are ranked and their most-different levels chosen, and an axis that runs out of transverse variation is split into finer conditional sub-axes. The recursion is what the name refers to, and it is the part that changes the asymptote rather than the constant.

### 5.1 The stack

Per accepted item, a bounded number of model calls regardless of corpus size:

1. **Spec choice.** Sample a pool of specs from the current axis lattice; embed their descriptions; keep the one with the largest residual outside the corpus's occupied eigenspace. Orthogonalization at the level of *conditions*, not outputs.
2. **Generation.** *K* parallel completions conditioned on the spec, with the spec's levels stated as **contracts** (behaviors the text must exhibit, not suggestions), plus an explicit avoid-list from the ledger.
3. **Judging.** One separate call scores craft (or, for exam items, validity), and checks each required behavior. Generation never grades itself.
4. **Selection.** Utility = 0.4·quality + 0.35·orthogonality + 0.25·capped gap, behind a typicality gate: a hard rejection of candidates lying beyond *z* running radii from the corpus centroid. The gate is negative supervision only (it may reject, never endorse, and passing it is not evidence of quality), and the gap term is **capped** at 2 running scales, so no candidate earns unbounded credit for sitting far from everything.
5. **Mining.** Every 25 accepts, show the model 50 sampled items and ask what they have in common. Append the answer to a JSONL ledger; a bounded slice becomes the next prompts' avoid-list.
6. **Refinement.** On a saturation signal, ask the generator to split the dominant saturated cell.

Every state the policy reads is bounded: running centroid and second-moment, EMA scales, a fixed ledger slice, a recent window plus a bounded random subsample of older items. The marginal cost of item 10,000 equals that of item 100.

### 5.2 Eliciting and refining latent behaviors

We ask the generator for axes that are near-orthogonal, structural rather than topical, and whose levels are all usable — explicitly steering away from subject matter, length, and rhyme toward craft decisions. On the live run this produced axes including *temporal stance*, *relationship to its own claim*, *syntactic weather*, and *the register the poem refuses*.

Refinement shows the model the saturated cell and the mined attractors and asks it either to split a level into finer sub-levels or to mint a new axis that applies only inside that cell. Refinement axes are **conditional** (they apply only when their parent level was chosen), so the lattice is a tree, and refining a saturated region does not inflate cost elsewhere.

### 5.3 Why "orthogonalize", not "randomize"

Forcing randomness (raising temperature, injecting random seed words) buys variance in the surface while leaving the mode structure intact, and degrades quality monotonically because temperature cannot distinguish *surprising* from *wrong*. The measurements bear this out: raising the sampling temperature to 1.6 moves distinct-2 from 0.3341 to 0.3342 and leaves 61.6% of the corpus byte-identical duplicates, against 72.3% at the default setting.

Forcing approximate orthogonality asks a different question (which direction is this corpus not yet spending energy on), which has a computable answer, improves rather than degrades quality when paired with a quality term, and stays informative as *n* grows. "Be random" gets no harder to satisfy and no more useful.

---

## 6. Experiments: 43,000 generations, three domains, seven methods

Generator: `openai/gpt-5.6-luna` via OpenRouter. Text embeddings: `nomic-embed-text` (768-d) served locally by Ollama, so embedding is free and the loop is never rate-limited by its own measuring instrument. Images: `gpt-image-1-mini`. Image embeddings: CLIP ViT-B/32. Audio: Lyria 3 Pro instrumentals. Audio embeddings: CLAP and MERT. Every corpus is append-only JSONL with embeddings checkpointed alongside, resumable after a kill.

### 6.1 Domains

**DALL·E instructions for post-modern artworks.** Diversity *is* the product: a clustered instruction set renders to a clustered image set. Chosen also because it lets us measure the same corpus in three spaces — words, text-embedding, and the rendered picture.

**Psychometric test questions.** Multiple-choice items assessing general reasoning. Two items with the same construct and the same surface are redundant at best and, in an operational bank, a validity problem.

**Instrumental-music prompts.** The longest chain in the paper: latent axes → a text prompt → a two-minute instrumental → an audio embedding. We render with Lyria 3 Pro (`lyria-3-pro-preview`), which returns roughly two minutes of audio per call in about 20 seconds and emits its own section plan (`[[A0]] [[B1]] [[C2]] [[D3]]`), a model-reported description of the form it chose that gives a discrete structural signal without waveform analysis. The generator proposed seven compositional axes for this domain — *formal trajectory*, *inter-layer rhythmic relationship*, *harmonic motion*, *timbral centre of gravity*, *density contour*, *pulse relationship*, *opening and closing frame* — craft decisions rather than genre labels, each expressible as something audible.

**Poetry.** A smaller pilot corpus, used to elicit and inspect the axis machinery on a domain where craft rather than content carries the variation, and to illustrate what the attractor mining recovers (§6.7).

**Prompt length.** A text-to-music model honours concrete, performable direction (named instruments, room character, register, articulation, rhythmic feel, how it ends), and ignores paragraphs of abstract compositional theory. Long prompts therefore inflate every text-side diversity metric while changing the audio far less, which is the proxy gap §6.6 measures. Prompts are held to 2–4 sentences (~300 characters target, 423–449 mean in the corpora below) of specifically audible, instrumental-only direction, and the audio metrics decide.

### 6.2 Methods compared

Five of these are real published approaches, implemented faithfully rather than as strawmen, each getting the same generator, the same embedder, and the same budget accounting as ours.

| arm | what it is |
|---|---|
| `naive` | plain repeated prompting — the honest floor |
| `high_temp` | temperature 1.6 — the trivial diversity lever |
| `self_instruct` | **Self-Instruct / Alpaca**: few-shot exemplars sampled from the pool, plus ROUGE-L ≤ 0.7 rejection against it |
| `evol_instruct` | **Evol-Instruct (WizardLM)**: sample an existing item, apply a random evolution operator (deepen, concretize, add constraint, harder reasoning, mutate form) |
| `persona` | **Persona-Hub / AttrPrompt**: a flat catalogue of personas × attributes, sampled uniformly |
| `rac` | **Recursive Axis Conditioning** (ours): recursively elicited axes, spec-level orthogonalization, capped repulsion behind a typicality gate, append-only attractor ledger |
| `rac+vision` | RAC, plus steering on the *rendered image* (§6.6) |

`persona` is the load-bearing comparison. It has language-valued latent conditioning (the same basic idea as ours), but no orthogonality selection, no ledger, and no recursive refinement. The gap between `persona` and `rac` isolates exactly what those three components buy.

One faithfulness caveat we chose deliberately: Self-Instruct's ROUGE filter compares a candidate against the entire pool, which is O(*n*) longest-common-subsequence computations per candidate and comes to dominate the loop at *n* in the thousands. We compare against a bounded random sample of 120 pool members. This makes our reimplementation *weaker* than the original at large *n*, and that is itself the point: the original's redundancy check does not have an infinite horizon, because its cost grows linearly in the corpus it is protecting.

### 6.3 Mode collapse is not a metaphor

Two 10,000-item corpora, one call each, no selection:

| domain | *n* | unique texts | exact-duplicate rate | most-repeated item |
|---|---|---|---|---|
| DALL·E instructions | 10,000 | 10,000 | 0.000 | 1× |
| psychometric items | 10,000 | ~1,656 | **0.723** | **1,637×** |

In the psychometric domain, 72.3% of a ten-thousand-item corpus is exact duplicate text, and one single question (*"What number comes next in the sequence: 2, 6, 12, 20, 30, ?"*) accounts for 1,637 of them. No embedding, no threshold, and no interpretation is required to see this; it is byte-identical repetition, and it is what a strong instruction-tuned model does when asked the same reasonable question ten thousand times.

Temperature barely helps: at *T* = 1.6 the duplicate rate is still 0.616. Conditioning nearly eliminates it: persona conditioning drops it to 0.002. Between those two numbers — randomness does not buy diversity, conditioning does.

It also shows the failure is domain-shaped and invisible from one vantage point. The identical pipeline, prompt style, and model produce zero duplicates on DALL·E instructions. A practitioner who validated their pipeline on the first domain and deployed it on the second would ship a bank that is three-quarters one question.

![Figure 3. Real gpt-5.6-luna corpora, n = 5 to 10,000. Literal measures (top row), and latent measures (bottom row) do not agree about what is happening.](figures/fig10_real_curves.png)

*Figure 3. Real gpt-5.6-luna corpora, n = 5 to 10,000. Literal measures (top row), and latent measures (bottom row) do not agree about what is happening.*


The same failure is visible in a corpus with no duplicates at all. The poetry pilot has an exact-duplicate rate of 0.000 under naive prompting and a distinct-2 of 0.610, so every literal counter reports a healthy corpus. Here are the opening lines of eight poems sampled at random from those sixty:

> At dawn, the river gathered up the stars
> At dusk, the windows gather fire,
> At dusk, the rooftops gather amber light,
> At dusk, the windows gather amber light,
> At dawn, the windows gather up the rain,
> At dusk, the river gathers every color
> At dusk, the river gathers up the day,
> At dusk, the river gathers up the sky,

Eight of eight are the same sentence with two slots filled. The closest pair in that corpus sits at cosine similarity 0.946 and shares its first line verbatim (*"At dusk, the windows gather gold,"*) before diverging into the same sparrow stitching the same thread across the same evening. Sixty samples from one prompt behave like two and a half distinct items by Vendi Score, and the duplicate counter sees none of it.

Eight openings from the sixty RAC produced under the same generator and budget:

> Had you arrived at the registry after midnight, when the lamps were still accountable,
> We have tried to piece that evening together from what remained.
> I am the conjurer in the green coat, stepping into the light.
> I have reconstructed the trench, Mara, from the surviving datum points.
> I wore the cobalt pressure suit, and the regolith took my weight,
> We went back through that winter in our minds.
> "Attend, O citizens, beneath the unappeased sky:
> At the appointed hour, you will rise beneath the bells,

Counterfactual conditional, collective reconstruction, first-person performance, a report addressed to a named absent person, plain retrospect, public proclamation, prophecy. The closest pair RAC produces sits at 0.809 and shares a subject rather than a template: one poem is a nested incantation about a coal inside a chamber inside a breast, the other a kitchen-table conjuring trick performed for a child in a red coat.

The measurements follow the reading. Against the naive corpus, RAC holds 2.7× the median nearest-neighbour distance (0.239 against 0.089), and cuts 4-gram self-repetition by a factor of 37 (0.0023 against 0.0855), at a distinct-2 of 0.774 against 0.610. Mean-centered Vendi moves very little in comparison, 38.81 against 37.00, and that is the honest reading of it: a spectral summary of sixty points in 768 dimensions is close to insensitive to the difference between sixty poems that begin the same way and sixty that do not. The nearest-neighbour statistic is what registers it, and the page registers it immediately.

### 6.4 Literal and latent diversity move in opposite directions

Measuring the DALL·E naive corpus as it grows from *n* = 5 to *n* = 10,000, in both spaces:

| *n* | distinct-2 | 4-gram self-repetition | *n*-gram Vendi | centered embedding Vendi |
|---|---|---|---|---|
| 5 | 0.870 | 0.031 | 4.9 | 3.79 |
| 40 | 0.492 | 0.211 | 30.2 | 24.13 |
| 300 | 0.252 | 0.384 | 142.5 | 55.70 |
| 1,000 | 0.145 | 0.520 | 246.6 | 63.05 |
| 1,750 | 0.112 | 0.577 | 283.3 | 65.30 |

Literal diversity collapses monotonically (by *n* = 1,750 more than half of each new instruction's 4-grams have already appeared), while latent diversity *rises* and then saturates. They are measurements of two different quantities that a single word, "diversity", has been covering for.

**Centering.** The uncentered embedding Vendi on this corpus reads 1.63 → 2.33, which would suggest ten thousand instructions behave like two distinct items. That number is mostly an artifact of the kernel. Same-domain embeddings sit in a narrow cone (mean pairwise cosine similarity is 0.883 here and 0.788 on the psychometric corpus), so the Gram spectrum is dominated by the shared mean direction and the score compresses toward 1. After removing the mean direction the same corpus reads 3.79 → 65.30, which is the honest curve and the one in the table. We report both, and we suggest any embedding-based diversity result publish the corpus's pairwise-similarity distribution alongside it, because the number is meaningless without it. Part II measured a mean pairwise cosine of 0.444 on its own pool with the same embedder — the cone is corpus-dependent, not a fixed property of the embedder, which is exactly why it has to be reported rather than assumed.

![Figure 4. The decoupling, normalised to each series' value at n = 5: literal diversity falls while latent diversity rises.](figures/fig11_literal_vs_latent.png)

*Figure 4. The decoupling, normalised to each series' value at n = 5: literal diversity falls while latent diversity rises.*


### 6.5 Competitive comparison at matched *n*

All seven arms, real corpora, DALL·E domain, every arm evaluated on the same number of accepted items (*n* = 200):

| arm | distinct-2 ↑ | self-repetition ↓ | *n*-gram Vendi ↑ | centered Vendi ↑ | median NN distance ↑ |
|---|---|---|---|---|---|
| naive | 0.293 | 0.345 | 107.6 | 50.45 | 0.045 |
| high temperature | 0.291 | 0.339 | 109.0 | 49.44 | 0.045 |
| Evol-Instruct | 0.386 | 0.353 | 108.1 | 50.10 | **0.021** |
| Self-Instruct | 0.545 | 0.131 | 140.4 | 67.45 | 0.121 |
| persona conditioning | 0.582 | 0.130 | 124.7 | 56.50 | 0.080 |
| **RAC** | 0.660 | 0.068 | 167.0 | 77.67 | 0.171 |
| **RAC + vision steering** | **0.699** | **0.040** | **170.9** | **91.97** | **0.192** |

RAC wins every column, and adding vision steering improves on the text-only variant by a further 18% in centered Vendi. Three of the baselines fail in different ways.

High temperature buys nothing here: distinct-2 of 0.291 against naive's 0.293, and an identical median nearest-neighbour distance of 0.045. Turning the temperature up moved the third decimal place, and moved it downward.

Evol-Instruct comes out worse than naive at the one thing a diversity method exists for. Its median nearest-neighbour distance is 0.021, against naive's 0.045 — it produces items *closer* to the existing corpus than independent sampling does. The mechanism is working as designed: mutating an existing item produces a neighbour of that item. Evol-Instruct raises *distinct-2* (0.386 vs 0.293), because the mutations reword things, so a purely literal evaluation would score it as an improvement. Measuring both levels is what catches it.

Self-Instruct's guard is literal, and it defends the literal level only. It achieves the best self-repetition of any baseline (0.131) (its ROUGE filter is doing real work), while its centered Vendi (67.45) trails RAC by 10 points and RAC with vision steering by 24. A ROUGE-L threshold cannot see semantic redundancy, and semantic redundancy is what remains once the lexical kind is filtered.

![Figure 5. Competitive comparison on real corpora at matched n, both domains, across four metrics.](figures/fig14_arms.png)

*Figure 5. Competitive comparison on real corpora at matched n, both domains, across four metrics.*


The psychometric domain repeats the ordering on all six arms at matched *n* = 1,058:

| arm | exact-dup ↓ | distinct-2 ↑ | self-repetition ↓ | *n*-gram Vendi ↑ | centered Vendi ↑ | median NN distance ↑ |
|---|---|---|---|---|---|---|
| naive | 0.668 | 0.114 | 0.855 | 11.8 | 7.1 | 0.000 |
| high temperature | 0.660 | 0.119 | 0.845 | 12.2 | 7.0 | 0.000 |
| Self-Instruct | 0.026 | 0.193 | 0.485 | 288.8 | 59.3 | 0.033 |
| Evol-Instruct | 0.025 | 0.224 | 0.470 | 307.4 | 75.0 | 0.047 |
| persona conditioning | 0.004 | 0.377 | 0.184 | 545.6 | 99.7 | 0.130 |
| **RAC** | **0.000** | **0.429** | **0.053** | **644.3** | **124.8** | **0.210** |

RAC wins every column here as well. The two undiversified arms score an order of magnitude worse on the latent measures because their duplicate rates of 0.66–0.67 mean the embedding metric is largely measuring repetition; deduplicating first raises them to 19.8 and 20.7, still far behind every conditioned method, and their median nearest-neighbour distance is exactly zero — the median item has a perfect twin.

### 6.6 The proxy problem: text diversity is nearly blind to image diversity

We rendered the first 199 DALL·E instructions from the naive corpus to actual images and embedded them with CLIP. The result is the most consequential measurement in this section:

> Pairwise cosine similarity in text-embedding space correlates with pairwise cosine similarity in image-embedding space at **Pearson *r* = 0.170**.

Knowing that two instructions are semantically far apart tells you very nearly nothing about whether the two pictures look different. Every text-side diversity method in the table above (ours included) is optimizing a proxy that explains roughly 2% of the variance in the thing the user actually receives.

The qualitative version is more damning than the correlation. Independently generated instructions, from a corpus with a 0.000 exact-duplicate rate and healthy lexical diversity, render to near-interchangeable pictures (Figure 7, top): the same magenta-and-cyan palette, a classical marble bust, neon signage, halftone collage, a receding grid. A vision judge shown samples of the set rates its distinctness 6–7 out of 10 and names the attractors precisely — *"muted beige, cream, brown, ochre and black foundations accented by saturated cyan/teal, turquoise, pink"*, *"appropriation of canonical or religious imagery, especially Mona Lisa-like female portraits"*, *"frontal, museum-like presentation with centered, symmetrical compositions"*.

This is a second mode collapse, downstream of ours, contributed by the image model and by the fact that much of what varies in the text ("post-modern", "appropriated source") lands in the same visual place.

![Figure 6. Text versus vision diversity on 97 rendered artworks. (c) Pairwise similarities in the two spaces correlate at only r = 0.170.](figures/fig12_vision.png)

*Figure 6. Text versus vision diversity on 97 rendered artworks. (c) Pairwise similarities in the two spaces correlate at only r = 0.170.*

![Figure 7. Sixteen renders from each policy, same generator and budget. Top row: naive prompting, a corpus with a 0.000 exact-duplicate rate and healthy lexical diversity that still collapses to one visual mode — magenta and cyan collage, classical bust, barcode, neon lettering. Bottom row: max-min steering with literal-space repulsion, where palette, medium, composition and register all vary. The bottom row remains fonder of grids and tiled panels than a human art director would be; §6.6 says why the objective does nothing to discourage that.](figures/fig13_contact_sheet.png)

*Figure 7. Sixteen renders from each policy, same generator and budget. Top row: naive prompting, a corpus with a 0.000 exact-duplicate rate and healthy lexical diversity that still collapses to one visual mode — magenta and cyan collage, classical bust, barcode, neon lettering. Bottom row: max-min steering with literal-space repulsion, where palette, medium, composition and register all vary. The bottom row remains fonder of grids and tiled panels than a human art director would be; §6.6 says why the objective does nothing to discourage that.*


A reader looking at the bottom row will notice that it still favours grids, panels and tiled compositions more than a human art director would. That is what the objective asks for and no more. A max-min corpus is scored on how far apart its items are, and nothing in the score says the corpus should look like any particular distribution of artworks — there is no likelihood term, no reference set, no penalty for sitting in a region of image space that real post-modern art rarely occupies. If the generator's prior happens to place a whole family of mutually distant images inside the tiled-grid region, spreading points is satisfied by staying there and varying what fills the cells. Part II's objective is the one that has a reference distribution in it, and a corpus scored on coverage of human-written material inherits a pull toward where that material actually sits. Buying both at once means carrying both terms, which is a different optimization from the one measured here.

The vision-steered arm closes the loop where the product actually lives. It renders a bounded sample of accepted instructions, embeds them with CLIP, and feeds two things back into the text-side loop: a least-squares map from the crowded *image* directions into instruction-embedding space, so the text-side orthogonality term can push away from visual redundancy it cannot itself perceive; and mined *visual* attractors from the vision judge, appended to the same append-only ledger as the textual ones and repelled against in subsequent prompts. It is the paper's mechanism applied one level down: the ledger already repels against what the model keeps saying, and now also against what it keeps showing. It is the best arm in the table.

### 6.7 Attractor mining works, and is legible

The mining step produces findings specific enough to act on. From the poetry pilot, round 1 (*n* = 25): *"self-correction and immediate retraction: speakers repeatedly interrupt their own claims"*; *"ceremonial or institutional address: the voice of a bell-ringer, town crier, registrar, witness"*; *"threshold imagery and delayed passage: doors, gates, windows, bridges, shores"*. By round 2 (*n* = 50) it tracks the corpus's *drift* rather than restating round 1, noting that technical vocabulary is now being placed inside mythic frames and that recursive epistemic backtracking has become structural.

These findings are nameable and therefore repellable, which is what makes them usable as prompt constraints; a finding of "similar tone" would not be. And several of them are artifacts of *our own axis elicitation* — asking for craft-level axes like "relationship to its own claim" reliably produces self-correcting speakers. **The system's own conditioning becomes the next attractor.** The ledger catches the system's own habits alongside the model's.

### 6.8 The exam bank: a judged enemy-item radius

Exam items invert the semantics of the other two domains. Two operational items closer than δ are *enemy items* (seeing one gives away the other) a test-security failure at any bank size, so the floor is a hard constraint rather than a term in a weighted sum, and the min-distance check must be exact against the full bank rather than subsampled. Item templates expose few manipulable slots, so each mode is a low-dimensional disk and the δ-packing number of a bank is finite and small: a bank has a capacity, and the operative question is what fraction of a nominal bank is actually usable. That turns on δ, so we measured it. 236 item pairs drawn from a human-written bank (MMLU) across the full range of embedding distance were put to a blind psychometric adjudication — does seeing one item give a material advantage on the other, through a shared fact, a re-skin, or matched distractor misconceptions — with the judge shown neither the distance nor the provenance. A logistic fit of enemy verdicts on distance crosses 50% at **δ = 0.0354**, which is the operative radius for this embedder and item type.

Applying that radius to real banks — six generation policies at 2,500 items each, and the human bank under identical treatment:

| bank | exact-dup | items with an enemy | usable after the floor |
|---|---|---|---|
| MMLU (human-written) | 0.024 | 0.092 | 2,373 (94.9%) |
| naive prompting | 0.695 | 0.884 | 382 (15.3%) |
| high temperature | 0.710 | 0.897 | 365 (14.6%) |
| Self-Instruct | 0.047 | 0.616 | 1,373 (54.9%) |
| Evol-Instruct | 0.012 | 0.424 | 1,870 (74.8%) |
| persona conditioning | 0.007 | 0.047 | 2,425 (97.0%) |
| **RAC** | **0.000** | **0.000** | **1,058 (100.0%)** |

A naively generated bank of 2,500 items yields 382 that can coexist on one form — 15% of nominal capacity, against 94.9% for the human bank. Temperature makes it slightly worse. Both conditioned policies exceed the human bank's usable fraction, and the axis-conditioned bank contains no enemy pair at all at the judged radius, though it is measured at 1,058 items rather than 2,500 and the comparison should be read at that size.
### 6.9 Head-to-head against a human-written exam bank

The comparisons above are against our own reimplementations, which is the right experiment for isolating mechanisms and the wrong one for the question *is this actually good?* — a reimplementation can be weak in ways that flatter us. So we also compare against an item bank that humans wrote: **MMLU**, ~14,000 multiple-choice items drawn from real practice exams and textbooks across 57 subjects, by many authors, with editorial review, over years. All arms are sampled uniformly at random and evaluated at matched *n* = 1,000 with identical metric code.

| source | exact-dup ↓ | distinct-2 ↑ | self-repetition ↓ | *n*-gram Vendi ↑ | centered Vendi ↑ | median NN dist ↑ |
|---|---|---|---|---|---|---|
| **MMLU (human-written)** | 0.0040 | **0.5888** | 0.0585 | 495.2 | **197.50** | **0.2838** |
| naive gpt-5.6-luna | 0.6560 | 0.1196 | 0.8383 | 12.1 | 6.52 | 0.0000 |
| high temperature | 0.6840 | 0.1161 | 0.8567 | 10.2 | 6.66 | 0.0000 |
| Self-Instruct | 0.0280 | 0.1990 | 0.4797 | 277.6 | 57.22 | 0.0351 |
| Evol-Instruct | 0.0060 | 0.2642 | 0.3855 | 381.9 | 89.23 | 0.0792 |
| persona conditioning | 0.0040 | 0.3887 | 0.1711 | 543.3 | 99.71 | 0.1346 |
| **RAC** | **0.0000** | 0.4377 | **0.0503** | **622.2** | 135.58 | 0.2235 |

The synthetic and human comparisons come out differently, and the difference matters.

Against every synthetic method, RAC wins every column. Centered Vendi 135.6 against the best baseline's 99.7 — a 36% margin over persona conditioning, which is the closest published relative of our approach and differs from it exactly by the three components we add. Self-repetition 0.050 against persona's 0.171. The undiversified arms are not close: naive and high-temperature prompting score ~6.5 centered Vendi, because two-thirds of their corpora are byte-identical duplicates, and their median nearest-neighbour distance is exactly 0.0000 — the median item has a perfect twin.

Against the human bank the result splits. We *beat* MMLU on exact duplication (0.0000 vs 0.0040 — the human bank has some), on 4-gram self-repetition (0.050 vs 0.059), and on *n*-gram Vendi (622 vs 495): our items reuse less language than human-written ones do. We *lose* on the semantic measures — centered Vendi 135.6 against 197.5, and median nearest-neighbour distance 0.224 against 0.284. We reach roughly 69% of a human exam bank's effective semantic diversity.

That gap is the honest measure of what is left. MMLU's spread comes from 57 genuinely different subjects; ours comes from an axis lattice a single model proposed in one call, refined a handful of times. The lexical result says our surface variety already exceeds human-authored items; the semantic result says our *conceptual* variety does not, and that closing the remaining 31% is a question about how much genuinely different subject matter the generator can be induced to reach, which is precisely the reachable-dimension question of §3, not a tuning problem.

### 6.10 Which metrics are independent of the objective

Our selection rule maximizes a weighted sum of embedding-space orthogonality and embedding-space min-gap. We then report embedding-space diversity metrics. Those two facts are not independent: the centered Vendi Score is a monotone function of how flat the embedding Gram spectrum is, which is close to exactly what the orthogonality term climbs, and the median nearest-neighbour distance *is* the min-gap term. To that extent, the wins on centered Vendi and median nearest-neighbour distance are partly tautological: the method optimizes them and the baselines do not, so they should be read as confirmation that the optimizer works rather than as independent evidence.

The independent evidence is that the method never observes the *literal* metrics at all. It reads embeddings; it has no access to token counts, *n*-gram overlap, or string identity. So exact-duplicate rate, distinct-2, 4-gram self-repetition, and *n*-gram Vendi are independent of the objective in a way the embedding metrics are not, and the method wins those too, including against the human-written bank on two of the four. Sorted by how much weight they can bear:

- **Embedding-space measures (Vendi, NN distance):** partly circular; confirmation that the optimizer works.
- **Literal-space measures (duplication, distinct-2, self-repetition, *n*-gram Vendi):** independent, since nothing in the method targets them. These carry the argument.
- **Rendered-artifact measures (CLIP, CLAP, MERT):** the most independent, living downstream of a second generative model the method never sees, which is why §6.6 matters more than its length suggests.

A reader who trusts only the third category still has the *r* = 0.170 result, which is a finding about every method in the table rather than a comparison between them.

---

### 6.11 Literal-space repulsion: fixing what the embeddings cannot see

Embedding metrics miss an entire class of repetition: tiled grids of one cell, prominent typography, a single shared palette, the same composition recolored. Audited with deliberately dumb, non-semantic signatures (a 16×16 luminance layout map, an autocorrelation tiling score, a hue histogram): half the images had a layout twin above 0.5 cosine, 40–45% were literal tilings, and 63–78% shared a palette, while prompt-level Jaccard sat at a healthy 0.14–0.17. Varied words, one visual mode: the conditional-dimension gap operating inside the *renderer*.

The fix is a four-quadrant repulsion ({text, vision} × {literal, latent}) where the two literal quadrants were previously unpopulated: content-word overlap penalties and live overused-word bans on the text side; and on the vision side, measured structural bans injected into prompts (grids banned when recent renders tile, dominant hue pairs named and banned, layout-change demands when layouts collide), plus a learned text→bad-structure bridge that penalizes candidates near prompts whose renders tiled. Rerun at matched budget, palette twins halved (0.78 → 0.35 in the coverage arm), the coverage objective improved 21% (0.206 → 0.247), and layout/tiling moved modestly (0.51 → 0.46–0.48; 0.45 → 0.33) — a partial improvement, with the residue attributable to the image model's own prior resisting text-side instruction.

### 6.12 Audio: prompts, the steered corpus, and embedder dependence

**Prompt-level diversity.** At matched *n*, matched prompt length and the same generator, the conditioned arm reaches centered Vendi 43.03 against naive's 22.99 (1.87×), halves 4-gram self-repetition (0.140 vs 0.291), raises distinct-2 (0.545 vs 0.348) and *n*-gram Vendi (75.3 vs 56.0), and holds roughly three times the room between nearest neighbours (0.079 vs 0.027), on 100 prompts per arm. Neither arm produces exact duplicates, so the effect is semantic. This replicates on a third artifact type the pattern of §6.3 and §6.5: conditioning moves the number, sampling temperature does not.

**The steered corpus.** 100 Lyria-3-Pro instrumentals generated by cross-modal steering with zero rejection — selection happens over prompts, every render is kept. Final measurements: 100% of tracks verify as instrumental in CLAP space (minimum instrumental-vs-vocal margin 0.077); mean-centered CLAP Vendi 17.43, against 11.5 (naive prompts), and 14.1 (axis-conditioned prompts) for the unsteered 47-track arms under the same embedder; opening loudness at 0.72 of each track's own median over the first three seconds, against a 0.46–0.61 baseline — the sparse-opening attractor substantially, not fully, suppressed.

**Structural control.** Lyria honors its own section-marker format (`[[A0]] [[B1]] …`), and ignores wall-clock timestamps. Compliance with a requested section sequence is length-dependent: exact at prompt lengths near 560 characters, 0.11 at a mean of 658, zero by ~1,600. Structural contracts for music must be short or they are noise.

Whether a prompt can be steered before rendering turns out to be a property of the embedder rather than of music. Prompt-to-track alignment (Spearman between prompt-pair similarity through the text tower and track-pair similarity through the audio tower, naive/conditioned arms):

| embedder | alignment | note |
|---|---|---|
| **MuQ-MuLan** (open MuLan-style) | **0.68 / 0.47** | a usable steering gradient |
| CLAP-fused | 0.50 / 0.21 | weak; ~0.05 on the steered corpus |
| CLAP-music | 0.18 / 0.06 | worst; median within-corpus NN distance 0.004 — it hears one track |
| MERT | — | no text tower (control) |

Cross-embedder agreement on pairwise track similarity spans 0.22–0.84; per-arm diversity verdicts flip between embedders, and each embedder nominates a different most-redundant pair. Two prescriptions follow: steer music with MuQ-MuLan rather than CLAP, and never publish an audio-diversity number without naming its embedder.

### 6.13 Scale and cost

The text corpora comprise 43,171 real generations for roughly $8.50 of OpenRouter spend, plus 690 rendered images across seven arms at two quality tiers, 100 rendered Lyria instrumentals, and 236 adjudicated exam-item pairs. Both `naive` arms reach *n* = 10,000; the reimplemented baselines reach 2,500 each. Every comparison is reported at a matched *n* that all compared arms actually reached.
---

## 7. Relation to the coverage problem

Two objectives share the generation budget. **Max–min** is a packing objective: *k*-center-like, driven by the worst-case nearest pair, with no approximation guarantee under greedy selection. **Coverage** (given a finite budget, maximize the volume of a union of ε-balls) is a covering objective, closer to facility location, and monotone submodular, so greedy selection carries a (1 − 1/*e*) guarantee. The sections above develop the max–min side; Part II develops the coverage side across two further domains.

Each objective's strength is the other's failure mode. Packing spreads points toward the boundary and over-invests in outliers, which is why max–min is sensitive to outliers: outliers are what it is designed to seek. Covering weights regions by measure and fills the bulk, which is why it leaves the frontier sparse. Neither reduces to the other, and the tools that serve one degrade the other.

Both problems share this paper's structural constraints, and we expect the conditional-dimension results to be *more* consequential for coverage than for packing: covered volume is governed by reachable dimension, not by how many distinct prompts one can write, so the gap between how many distinct specs one can write and how many independent directions they actually move in bites harder there. The no-inverse-oracle assumption also degrades coverage's greedy guarantee, since the guarantee is relative to the best subset *of what was proposed*, and the proposal distribution is confined to a slice.

---

## 8. Limitations

**One embedding oracle.** All geometry is `nomic-embed-text` geometry in the live runs. Whether "diverse" under one embedder transfers to another is untested here and is a real threat to any embedding-based diversity claim, including ours.

**Judge bias is unmeasured.** Language-model judges have systematic preferences, often *for* fluent and typical text, which is exactly the bias that would work against a diversity system. We use a judge for craft and validity scores and did not calibrate its error against human raters.

**Additive spec composition.** The axis scoring treats conditioning attributes as composing additively in embedding space, which is what makes a spec's position predictable before it is generated. Real interactions between prompt attributes are not additive.

**The exam-item judgments are model-made.** The enemy-item radius is calibrated against a blind adjudication by a language model under a psychometric rubric (§6.8), not by a credentialed psychometrician, and δ is specific to this embedder and this item type.

**Quality is scalar.** Craft is not one number, and collapsing it to one lets a system trade away dimensions of quality the judge does not score.

---

## 9. Conclusion

The infinite-horizon diversity objective is easy to write, cheap to maintain, and largely beside the point. What determines whether a corpus can keep growing without collapsing into paraphrase is the dimension of the generator's conditional support relative to the manifold it lives in, and every practically important consequence follows from that one gap: saturation arrives at rate *n*<sup>−1/*m*</sup> rather than *n*<sup>−1/*d*</sup>; a single prompt covers an ε<sup>*d*−*m*</sup> fraction of what the model could write; and only transverse prompt motion raises the ceiling.

Given an embedding oracle and no inverse, the system cannot compute its way to the next item. It can only choose what to condition on. The axis-scoring rule (spread, transversality, independence, headroom) is our answer to that choice, and its most valuable output is not the ranking but the **refine** signal: the moment it reports that nothing available is transverse any more is the moment the horizon has actually been reached, and the only remaining move is to ask the generator to subdivide its own vocabulary of variation.

That recursion is the single mechanism we found that changes the asymptote rather than the constant. Everything else (better selection, more candidates, higher temperature) buys a constant factor against a problem that is asymptotic, and the loudest of them buys it by quietly breaking the generator.

The measurements make the stakes concrete in a way the theory could not. A strong model, asked a perfectly reasonable question ten thousand times, returns the same item 1,637 times; the temperature knob barely moves that number and conditioning nearly erases it. Two diversity metrics computed on the same growing corpus point in opposite directions. And the text embeddings every method in this literature optimizes turn out to explain about two percent of the variance in whether the rendered images look alike. Each of those is a reason to distrust a single diversity number, and together they are the argument for the practice we ended up recommending: measure at the level of the artifact you are shipping, report the literal and the latent separately, publish the similarity distribution your kernel is operating on, and count your duplicates before you compute anything else.

---
