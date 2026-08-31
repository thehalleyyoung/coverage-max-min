# Coverage and Max-Min Diversity in Synthetic Data Generation

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
consequences: novelty at fixed prompt decays as $n^{-1/m}$, not
$n^{-1/d}$; one prompt ε-covers a vanishing $ε^{d-m}$
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

# Part I — Max-min diversity: no two items alike


### Abstract

We study the problem of generating a corpus that stays diverse as it grows without bound. The objective is a convex combination of two terms evaluated against everything generated so far: the mean distance from a new item's embedding to the existing corpus, which we want small because it keeps items typical and on-manifold, and the minimum distance to the existing corpus, which we want large because it keeps items non-redundant. We call the resulting problem *infinite-horizon diversity*, and we are interested specifically in its asymptotics: what happens to the objective, and to the cost of satisfying it, as *n* grows.

Our central claim locates the interesting structure in the **support** rather than in the objective. Conditioned on a fixed prompt *x*, a language model's output distribution concentrates on a submanifold of dimension *m* that is far smaller than the dimension *d* of the reachable semantic manifold. We prove and numerically verify four consequences of that gap. Novelty at fixed prompt decays like $n^{-1/m}$, not $n^{-1/d}$, so saturation arrives much sooner than the ambient dimension suggests (measured slopes −0.472, −0.309, −0.195 for *m* = 2, 3, 5 against predictions −0.5, −0.333, −0.2). A single prompt ε-covers a vanishing $ε^{d-m}$ fraction of the manifold. Only prompt motion *transverse* to the already-occupied span raises the reachable dimension: parallel motion leaves the union's effective dimension at 1.9, transverse motion takes it to 11.2. Under free prompt-switching the optimal number of samples per prompt is exactly one; an interior optimum appears only once switching has overhead (*n*\* = 10 at cost *c* = 3, *n*\* = 30 at *c* = 30).

We assume an embedding oracle mapping text to a fixed-size vector, and — critically — **no reverse oracle**. We can compute exactly where we would like the next item to land and still have no way to decode that point into text. Every architectural choice follows from that asymmetry: the system must propose, measure, and select rather than solve, and the only steering handles are language-valued.

We therefore build the generation stack (**Recursive Axis Conditioning**, RAC) around latent behaviors elicited from the generator itself, organized by an **axis-scoring rule** that ranks candidate latent variables by the product of four measurable factors (spread, transversality, independence, and headroom), and selects each variable's most-different values by farthest-point search in level space. When the score reports that no available variable clears a promise floor, it emits a *refine* decision, and the generator is asked to split the exhausted variable into finer sub-variables. The recursion is the only mechanism in the paper that changes the asymptote rather than the constant.

We test this on **43,000 real generations** from `openai/gpt-5.6-luna` across three domains — DALL·E instructions for post-modern artworks, psychometric exam items, and instrumental-music prompts — measuring diversity at three levels: literal (*n*-gram), latent (text embedding), and, where the artifact is not text, in the space the product actually occupies (CLIP for rendered images, CLAP and MERT for rendered audio). Four results stand out. Asked the same reasonable question ten thousand times, the model returns a psychometric corpus that is **72.3% exact duplicates**, one item repeated 1,637 times; raising the temperature to 1.6 leaves it at 61.6%, while latent conditioning drops it to 0.2%. Literal and latent diversity move in **opposite directions** as the corpus grows. Text-embedding similarity predicts *rendered-image* similarity at only **r = 0.170**, so every text-side diversity method (ours included) optimizes a proxy explaining about 3% of the variance in what the user receives. And against five real published methods (Self-Instruct, Evol-Instruct, Persona-Hub, high-temperature, naive) our method wins every metric, beating the strongest by 36% in mean-centered Vendi; against **human-written MMLU** it wins on lexical diversity and self-repetition while reaching 69% of a human exam bank's semantic diversity.

---

### I.1 Introduction

Ask a language model for a poem ten times and you get ten poems. Ask it ten thousand times and you do not get ten thousand poems; you get a few hundred poems and a great deal of paraphrase. Any fixed conditional distribution behaves this way under repeated sampling, whatever model supplies it. The distribution has a shape, and repeated sampling traces that shape ever more densely rather than expanding it.

The practical version of this problem is now everywhere. Synthetic training data is valuable in proportion to how much it is *not* already in the training set. Evaluation suites that cluster in a few families give false assurance. Test-item banks, red-team prompt sets, persona corpora, and augmentation pipelines all consume generated text in bulk, and all of them degrade in a way that standard quality metrics do not detect: every individual item is fine, and the collection is redundant.

The natural formalization is a two-term objective. Let *E* be an embedding oracle and let *X<sub>n</sub>* = {*x*<sub>1</sub>, …, *x<sub>n</sub>*} be the corpus so far. For a candidate *c*,

$$
J_\lambda(c \mid X_n) \;=\; \lambda \cdot \underbrace{\frac{1}{n}\sum_{i=1}^{n}\lVert E(c) - E(x_i)\rVert}_{\text{anchor: keep it typical}} \;-\; (1-\lambda)\cdot \underbrace{\min_{i \le n} \lVert E(c) - E(x_i)\rVert}_{\text{repulsion: keep it new}}
$$

and we accept the candidate minimizing *J*<sub>λ</sub>. The anchor term keeps generation on the manifold; the repulsion term pushes it away from what already exists. The interesting question is what happens to this as *n* → ∞.

Our answer is that the objective is not where the difficulty lives. Both terms are cheap to maintain online and both behave predictably. The difficulty is that **the set of things the generator can propose is much smaller than it appears**, and the objective is powerless to enlarge it. Selection can only choose among what is offered. If the offering is confined to a thin slice, no selection rule (however clever, however well-tuned its λ) can produce diversity that the proposal distribution does not contain. Worse, as we show, aggressive selection under a fixed proposal distribution systematically degrades quality, because the only way to keep winning a novelty contest inside an exhausted region is to leave the manifold.

#### I.1.1 The oracle asymmetry

We assume throughout:

> **Assumption (embedding oracle, no inverse).** We have access to *E*: Text → $ℝ^{D}$, computable on demand. We have **no** $E^{-1}$: $ℝ^{D}$ → Text.

This is the structural fact that shapes the entire design, and it is worth dwelling on because it is easy to forget when writing down objectives. Given *X<sub>n</sub>* we can compute, exactly and cheaply, the point in $ℝ^{D}$ that would most improve any of our diversity measures — the direction of least occupied spectral energy, the centre of the largest empty ball, the point maximizing marginal coverage. Knowing that point is worth nothing on its own. There is no procedure that turns a target embedding back into a poem.

Consequently the system cannot *solve* for its next item. It can only:

1. **propose** (sample from *p*(· | *x*) for some prompt *x* we are able to write;
2. **measure**) embed the proposals and score them;
3. **select** — keep one.

All of the control we have is therefore in step 1, in the choice of *x*, and that choice is expressible only in language. This is why the latent variables in our system are *language-valued* (named axes with named levels), rather than continuous codes: they are the only handles that reach the generator. It is also why the axis scoring of §I.4 exists. **The axis scoring is a surrogate for the missing inverse.** Since we cannot decode the direction we want to travel, we instead score the language-valued conditions we *can* write by how nearly their induced output distributions point that way.

#### I.1.2 Contributions

- **A conditional-dimension theory of generative saturation** (§I.3). Four theorems relating the conditional support dimension *m* to the reachable manifold dimension *d*, each verified numerically. The practically important one is that saturation is governed by *m*, which is small, and not by *d*.
- **An axis-scoring rule** (§I.4). A decision procedure for which latent variable to condition on next and which of its values to use, together with a principled termination/recursion signal.
- **A budget-matched evaluation protocol** (§I.6). Every method receives the same generator, the same seed material, and the same number of generator calls. This is the fair frame for selection-based diversity methods, which buy diversity by discarding.
- **Three domains with different diversity semantics** (§I.6): instructions for image generation and prompts for instrumental music, where diversity is a soft objective, and psychometric exam items, where two items closer than δ are a test-security failure and the floor is hard.
- **43,000 real generations** on `openai/gpt-5.6-luna` with locally-computed embeddings, across three domains and seven methods (§I.6).

---

### I.2 Related work

**Diversity metrics.** The Vendi Score (Friedman & Dieng, 2022) is the exponential of the von Neumann entropy of a normalized similarity matrix, interpretable as an effective number of distinct items. We use it throughout, with a computational note: under a linear kernel on unit-normalized embeddings, the *n* × *n* Gram matrix and the *D* × *D* second-moment matrix share their nonzero spectrum, so the score costs *O*(*nD*² + *D*³), rather than *O*(*n*³). We verify this to machine precision (max eigenvalue mismatch 1.1 × 1$0^{-16}$). This matters for our setting specifically: an infinite-horizon method needs an evaluation whose cost does not grow superlinearly in the thing it is evaluating.

**Selection and subset choice.** Determinantal point processes model repulsion via volume in feature space. Farthest-point / *k*-center greedy is the classical max–min packing heuristic and we use it as a baseline (`fps_post`) and, at the level of axis values, as a component (§I.4.3). SemDeDup-style near-duplicate removal at a fixed radius appears as `dedup_post`. All of these are *post hoc* selectors over a fixed pool, which is exactly their limitation in our setting: they cannot change what is in the pool.

**Quality-diversity.** MAP-Elites and its descendants maintain an archive indexed by hand-designed behavior descriptors, seeking an elite per cell. Our axis lattice is a close relative, with two differences that matter here: the descriptors are *elicited from the generator rather than designed by us*, and the lattice *refines itself* when a cell saturates, so the archive's resolution is not fixed in advance.

**Mode collapse in instruction-tuned models.** Repeated sampling from aligned models is well documented to produce low lexical and semantic entropy relative to base models. We take this as our empirical starting point and model it explicitly as skewed mixture weights over style modes.

**Novelty search.** Novelty search in evolutionary computation rewards behavioral distance from an archive. Our repulsion term is a direct analogue, operating over an archive of embeddings rather than of behavior descriptors.

**Coverage.** Part II treats the covering objective (maximizing the volume of a union of ε-balls at a *finite* budget), which is submodular where max–min is not. §I.7 relates the two.

---

### I.3 Theory: conditional dimension governs saturation

#### I.3.1 Setup

Let *M* ⊂ $ℝ^{D}$ be the reachable semantic manifold, dim *M* = *d*: the set of embeddings of texts the generator could produce under *some* prompt. For a fixed prompt *x*, let *S<sub>x</sub>* ⊆ *M* be the support of *E*<sub>#</sub>*p*(· | *x*), with dim *S<sub>x</sub>* = *m*.

The empirical claim behind everything below is that **_m_ ≪ _d_**. A prompt fixes topic, stance, form, register, and rhetorical strategy; what remains free is a low-dimensional residue — some lexical choice, some ordering, some surface variation. Our live pilot supports this directly: 60 poems generated from an identical naive prompt have a median nearest-neighbour cosine similarity of 0.911 and a Vendi Score of 2.55, meaning sixty samples behave like roughly two and a half distinct items.

Throughout, *B*(*x*, *r*) is the ball of radius *r* and we write *g<sub>n</sub>* for the min-gap of a fresh draw against a corpus of size *n*.

#### I.3.2 Fixed-prompt saturation

> **Theorem 1.** Let *X<sub>n</sub>* be *n* i.i.d. draws from a distribution with density bounded above and below on an *m*-dimensional manifold *S*. For a fresh draw *c* from the same distribution, 𝔼[*g<sub>n</sub>*] = Θ($n^{-1/m}$).

*Proof sketch.* ℙ(*g<sub>n</sub>* > *r*) = ℙ(no sample in *B*(*c*, *r*)) = (1 − μ(*B*(*c*, *r*))$)^{n}$, and μ(*B*(*c*, *r*)) = Θ($r^{m}$) by density bounds on an *m*-manifold. So ℙ(*g<sub>n</sub>* > *r*) ≈ exp(−*n*Θ($r^{m}$*)), which transitions at *r* = Θ($n^{-1/m}$); integrating the tail gives the expectation. ∎

The content is the exponent. With *d* = 32, a naive reading predicts $n^{-1/32}$ — essentially flat, novelty never exhausted. The true rate at *m* = 3 is $n^{-1/3}$, which is roughly a 10× loss of headroom per 1000× of corpus. Measured slopes: −0.472 (*m* = 2), −0.309 (*m* = 3), −0.195 (*m* = 5), against −0.5, −0.333, −0.2. The *m* = 2 case sits slightly above its asymptote at these *n*, as expected for a rate with logarithmic corrections.

> **Corollary 1.1 (oversampling is not a strategy).** Drawing *K* candidates and keeping the most novel multiplies the expected gap by Θ($K^{1/m}$) at best. To recover the headroom lost by a 1000× larger corpus, *K* must grow by 1000.

Verified at *m* = 3: best-of-8 yields a gain of 2.19 against $K^{1/3}$ = 2.00. Selection is a constant-factor fix to an asymptotic problem.

#### I.3.3 The slice deficit

> **Theorem 2.** If *m* < *d*, then vol<sub>*d*</sub>(*S<sub>x</sub>*) = 0, and the fraction of *M* within ε of *S<sub>x</sub>* is Θ($ε^{d-m}$).

*Proof sketch.* The ε-neighbourhood of an *m*-dimensional set in *d* dimensions is a tube of volume Θ($ε^{d-m}$) · vol<sub>*m*</sub>(*S<sub>x</sub>*) for ε below the reach of *S<sub>x</sub>*. Dividing by vol<sub>*d*</sub>(*M*) gives the claim. ∎

Measured at *d* = 8, *m* = 2: fitted exponent 5.06 against a predicted 6, with covered fractions falling 0.051 → 0.0001 as ε goes 0.5 → 0.12. The fitted exponent sits below prediction because at the larger ε the tube is no longer thin relative to the manifold, which flattens the low-ε end of the fit; the qualitative claim (coverage collapsing polynomially in ε with a large exponent) is unambiguous.

The practical reading is severe. **A single prompt, sampled infinitely often, covers essentially none of what the model could write.** Not "less than we would like" — a fraction that goes to zero polynomially as the resolution of interest sharpens.

#### I.3.4 Transversality: which prompt motion helps

Let a prompt schedule induce slices *S*<sub>1</sub>, …, *S<sub>P</sub>*. Write *V*<sub>occ</sub> for the span of the corpus's dominant directions.

> **Theorem 3.** dim(⋃<sub>*j*</sub> *S<sub>j</sub>*) = min(*d*, *m* + rank{*c<sub>j</sub>* − *c*<sub>1</sub>}), where *c<sub>j</sub>* is the centre of *S<sub>j</sub>*. In particular, displacements lying inside span(*S*<sub>1</sub>) contribute nothing to the union's dimension.

*Proof sketch.* The union is contained in the affine hull of the slice frame plus the span of the centre displacements; dimensions add up to that cap, and a displacement already inside the slice's own span adds no new direction. ∎

Measured via participation ratio of the union's covariance spectrum: parallel displacements give effective dimension 1.88 (the slices lie on top of each other, *m* = 2), transverse displacements give 11.18.

![Figure 1. Measured saturation. (a) Expected min-gap of a fresh draw decays as n^(-1/k), with fitted slopes matching theory for k = 3, 8, 32. (b) Best-of-K oversampling buys only K^(1/k).](figures/fig5_scaling.png)

*Figure 1. Measured saturation. (a) Expected min-gap of a fresh draw decays as n^(-1/k), with fitted slopes matching theory for k = 3, 8, 32. (b) Best-of-K oversampling buys only K^(1/k).*


This is the theorem that makes the axis scoring of §I.4 possible. It measures the value of conditioning on a new latent variable as **how much of its variation lies outside what the corpus already spans** — a computable quantity, in place of the intuition of how different it sounds.

#### I.3.5 Breadth versus depth

Given budget *B*, split as *P* prompts × *n* samples each, with per-prompt switching cost *c* (eliciting a spec, embedding it, occasionally refining), so *B* = *P*(*c* + *n*).

> **Theorem 4.** With *c* = 0 the coverage-optimal depth is *n*\* = 1. For *c* > 0 the optimum is interior and increases with *c*.

Measured: with free switching, depth 1 achieves coverage 0.995 while spending the entire budget on one prompt achieves 0.091 — a 10.9× difference. With *c* = 3, *n*\* = 10; with *c* = 30, *n*\* = 30.

![Figure 2. Breadth versus depth at fixed budget. With free prompt-switching the optimum is one sample per prompt; an interior optimum appears only once switching is priced.](figures/fig8_breadth_depth.png)

*Figure 2. Breadth versus depth at fixed budget. With free prompt-switching the optimum is one sample per prompt; an interior optimum appears only once switching is priced.*


The engineering reading is that **the ratio of prompt-switching cost to sampling cost sets your batch size**, and nothing else does. If constructing a new spec is as cheap as a generation, generate one item per spec.

#### I.3.6 What this means for the objective

Returning to *J*<sub>λ</sub>: the anchor term is exactly computable from two running statistics. Writing μ*<sub>n</sub>* for the corpus centroid,

$$
\frac{1}{n}\sum_i \lVert c - x_i\rVert^2 = \lVert c - \mu_n\rVert^2 + \frac{1}{n}\sum_i \lVert x_i - \mu_n\rVert^2
$$

verified to zero relative error. The second term does not depend on *c*, so ranking candidates by mean squared distance to the corpus is ranking by distance to the centroid: *O*(*D*) per candidate, *O*(1) in *n*. The repulsion term needs a nearest-neighbour query, which we bound by subsampling. Neither term is the bottleneck. **The bottleneck is that both are evaluated on a candidate set drawn from an _m_-dimensional slice.**

---

### I.4 Scoring candidate axes

#### I.4.1 The question the axis scoring answers

Theorem 3 says progress requires moving the prompt transversely to what the corpus already spans. Theorem 4 says move often. Together they pose a concrete question at every step: **among the latent variables we could condition on next, which one moves the slice most transversely, per unit of budget?** Since we have no inverse oracle, this is the closest available substitute for computing the ideal next item.

Each candidate axis *a* has language-valued levels *L*(*a*). We embed the level descriptions and read off four quantities.

#### I.4.2 The four factors

**Spread** — mean pairwise distance between *a*'s level embeddings. Are this axis's values actually different *from each other*? An axis whose levels are near-synonyms cannot separate outputs however it is sampled. This catches the most common failure of LLM-proposed axes: plausible-sounding dimensions whose levels all mean the same thing.

**Transversality** — the fraction of *a*'s level-to-level variation lying outside the corpus's occupied eigenspace (top eigenvectors of the second-moment matrix carrying 80% of spectral energy). This is Theorem 3 made into a number. An axis whose levels differ only along directions the corpus already spans adds density, not reach.

**Independence** — 1 − max principal-angle cosine between *a*'s variation subspace and those of the axes **already in use**. An axis that restates an active axis is redundant however transverse it looks alone. Independence is measured against the active set and never against rival candidates: two equally good candidates would otherwise drive each other's score to zero and rank below a useless axis with no near-twin. Redundancy *among* candidates is a selection problem, handled by greedy selection that re-scores after every pick, so a duplicate is penalized at the point where the redundancy comes into existence.

**Headroom** — a normalized entropy deficit of how unevenly the corpus has sampled *a*'s levels, plus the fraction of levels never used. This is the only time-varying factor; it is what makes the ranking change as generation proceeds rather than being a fixed property of the axis set. An axis can be excellent and already spent.

The **promise** of an axis is the product:

$$
\mathrm{promise}(a) = \mathrm{spread}(a)\cdot\mathrm{transversality}(a)\cdot\mathrm{independence}(a)\cdot\mathrm{headroom}(a)
$$

Multiplicative, not additive, and deliberately so: an axis needs all four, and any one at zero should zero the score rather than be averaged away by the others. A high-spread axis with zero transversality is a distinction without a difference; a perfect axis with zero headroom is already spent.

#### I.4.3 Choosing values, and recursing

Once an axis is chosen we do not sample its levels uniformly. We take the **max–min subset** of its level embeddings by greedy farthest-point search, seeded deterministically at the level furthest from the level centroid — the packing problem again, one level down.

When the best available promise falls below a floor, the axis scoring emits `decision: "refine"` instead of `"condition"`. This is the **exhaustion signal**, and it is the honest one: it says the current lattice has nothing transverse left to offer, and no amount of further sampling will change that. The system then asks the generator to split the exhausted level into finer sub-levels (§I.5.2), producing a child axis scored by this same axis scoring. On the ground-truth test with only in-span axes available, the axis scoring correctly returns `refine`.

Cost is *O*(|*A*| · *D*²) per decision and does not grow with *n*.

---

### I.5 Method

We call the method **Recursive Axis Conditioning** (RAC): the generator is asked for the language-valued axes along which its own outputs can differ, those axes are ranked and their most-different levels chosen, and an axis that runs out of transverse variation is split into finer conditional sub-axes. The recursion is what the name refers to, and it is the part that changes the asymptote rather than the constant.

#### I.5.1 The stack

Per accepted item, a bounded number of model calls regardless of corpus size:

1. **Spec choice.** Sample a pool of specs from the current axis lattice; embed their descriptions; keep the one with the largest residual outside the corpus's occupied eigenspace. Orthogonalization at the level of *conditions*, not outputs.
2. **Generation.** *K* parallel completions conditioned on the spec, with the spec's levels stated as **contracts** (behaviors the text must exhibit, not suggestions), plus an explicit avoid-list from the ledger.
3. **Judging.** One separate call scores craft (or, for exam items, validity), and checks each required behavior. Generation never grades itself.
4. **Selection.** Utility = 0.4·quality + 0.35·orthogonality + 0.25·capped gap, behind a typicality gate: a hard rejection of candidates lying beyond *z* running radii from the corpus centroid. The gate is negative supervision only (it may reject, never endorse, and passing it is not evidence of quality), and the gap term is **capped** at 2 running scales, so no candidate earns unbounded credit for sitting far from everything.
5. **Mining.** Every 25 accepts, show the model 50 sampled items and ask what they have in common. Append the answer to a JSONL ledger; a bounded slice becomes the next prompts' avoid-list.
6. **Refinement.** On a saturation signal, ask the generator to split the dominant saturated cell.

Every state the policy reads is bounded: running centroid and second-moment, EMA scales, a fixed ledger slice, a recent window plus a bounded random subsample of older items. The marginal cost of item 10,000 equals that of item 100.

#### I.5.2 Eliciting and refining latent behaviors

We ask the generator for axes that are near-orthogonal, structural rather than topical, and whose levels are all usable — explicitly steering away from subject matter, length, and rhyme toward craft decisions. On the live run this produced axes including *temporal stance*, *relationship to its own claim*, *syntactic weather*, and *the register the poem refuses*.

Refinement shows the model the saturated cell and the mined attractors and asks it either to split a level into finer sub-levels or to mint a new axis that applies only inside that cell. Refinement axes are **conditional** (they apply only when their parent level was chosen), so the lattice is a tree, and refining a saturated region does not inflate cost elsewhere.

#### I.5.3 Why "orthogonalize", not "randomize"

Forcing randomness (raising temperature, injecting random seed words) buys variance in the surface while leaving the mode structure intact, and degrades quality monotonically because temperature cannot distinguish *surprising* from *wrong*. The measurements bear this out: raising the sampling temperature to 1.6 moves distinct-2 from 0.3341 to 0.3342 and leaves 61.6% of the corpus byte-identical duplicates, against 72.3% at the default setting.

Forcing approximate orthogonality asks a different question (which direction is this corpus not yet spending energy on), which has a computable answer, improves rather than degrades quality when paired with a quality term, and stays informative as *n* grows. "Be random" gets no harder to satisfy and no more useful.

---

### I.6 Experiments: 43,000 generations, three domains, seven methods

Generator: `openai/gpt-5.6-luna` via OpenRouter. Text embeddings: `nomic-embed-text` (768-d) served locally by Ollama, so embedding is free and the loop is never rate-limited by its own measuring instrument. Images: `gpt-image-1-mini`. Image embeddings: CLIP ViT-B/32. Audio: Lyria 3 Pro instrumentals. Audio embeddings: CLAP and MERT. Every corpus is append-only JSONL with embeddings checkpointed alongside, resumable after a kill.

#### I.6.1 Domains

**DALL·E instructions for post-modern artworks.** Diversity *is* the product: a clustered instruction set renders to a clustered image set. Chosen also because it lets us measure the same corpus in three spaces — words, text-embedding, and the rendered picture.

**Psychometric test questions.** Multiple-choice items assessing general reasoning. Two items with the same construct and the same surface are redundant at best and, in an operational bank, a validity problem.

**Instrumental-music prompts.** The longest chain in the paper: latent axes → a text prompt → a two-minute instrumental → an audio embedding. We render with Lyria 3 Pro (`lyria-3-pro-preview`), which returns roughly two minutes of audio per call in about 20 seconds and emits its own section plan (`[[A0]] [[B1]] [[C2]] [[D3]]`), a model-reported description of the form it chose that gives a discrete structural signal without waveform analysis. The generator proposed seven compositional axes for this domain — *formal trajectory*, *inter-layer rhythmic relationship*, *harmonic motion*, *timbral centre of gravity*, *density contour*, *pulse relationship*, *opening and closing frame* — craft decisions rather than genre labels, each expressible as something audible.

**Poetry.** A smaller pilot corpus, used to elicit and inspect the axis machinery on a domain where craft rather than content carries the variation, and to illustrate what the attractor mining recovers (§I.6.7).

**Prompt length.** A text-to-music model honours concrete, performable direction (named instruments, room character, register, articulation, rhythmic feel, how it ends), and ignores paragraphs of abstract compositional theory. Long prompts therefore inflate every text-side diversity metric while changing the audio far less, which is the proxy gap §I.6.6 measures. Prompts are held to 2–4 sentences (~300 characters target, 423–449 mean in the corpora below) of specifically audible, instrumental-only direction, and the audio metrics decide.

#### I.6.2 Methods compared

Five of these are real published approaches, implemented faithfully rather than as strawmen, each getting the same generator, the same embedder, and the same budget accounting as ours.

| arm | what it is |
|---|---|
| `naive` | plain repeated prompting — the honest floor |
| `high_temp` | temperature 1.6 — the trivial diversity lever |
| `self_instruct` | **Self-Instruct / Alpaca**: few-shot exemplars sampled from the pool, plus ROUGE-L ≤ 0.7 rejection against it |
| `evol_instruct` | **Evol-Instruct (WizardLM)**: sample an existing item, apply a random evolution operator (deepen, concretize, add constraint, harder reasoning, mutate form) |
| `persona` | **Persona-Hub / AttrPrompt**: a flat catalogue of personas × attributes, sampled uniformly |
| `rac` | **Recursive Axis Conditioning** (ours): recursively elicited axes, spec-level orthogonalization, capped repulsion behind a typicality gate, append-only attractor ledger |
| `rac+vision` | RAC, plus steering on the *rendered image* (§I.6.6) |

`persona` is the load-bearing comparison. It has language-valued latent conditioning (the same basic idea as ours), but no orthogonality selection, no ledger, and no recursive refinement. The gap between `persona` and `rac` isolates exactly what those three components buy.

One faithfulness caveat we chose deliberately: Self-Instruct's ROUGE filter compares a candidate against the entire pool, which is O(*n*) longest-common-subsequence computations per candidate and comes to dominate the loop at *n* in the thousands. We compare against a bounded random sample of 120 pool members. This makes our reimplementation *weaker* than the original at large *n*, and that is itself the point: the original's redundancy check does not have an infinite horizon, because its cost grows linearly in the corpus it is protecting.

#### I.6.3 Mode collapse is not a metaphor

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

#### I.6.4 Literal and latent diversity move in opposite directions

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


#### I.6.5 Competitive comparison at matched *n*

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

#### I.6.6 The proxy problem: text diversity is nearly blind to image diversity

We rendered the first 199 DALL·E instructions from the naive corpus to actual images and embedded them with CLIP. The result is the most consequential measurement in this section:

> Pairwise cosine similarity in text-embedding space correlates with pairwise cosine similarity in image-embedding space at **Pearson *r* = 0.170**.

Knowing that two instructions are semantically far apart tells you very nearly nothing about whether the two pictures look different. Every text-side diversity method in the table above (ours included) is optimizing a proxy that explains roughly 2% of the variance in the thing the user actually receives.

The qualitative version is more damning than the correlation. Independently generated instructions, from a corpus with a 0.000 exact-duplicate rate and healthy lexical diversity, render to near-interchangeable pictures (Figure 7, top): the same magenta-and-cyan palette, a classical marble bust, neon signage, halftone collage, a receding grid. A vision judge shown samples of the set rates its distinctness 6–7 out of 10 and names the attractors precisely — *"muted beige, cream, brown, ochre and black foundations accented by saturated cyan/teal, turquoise, pink"*, *"appropriation of canonical or religious imagery, especially Mona Lisa-like female portraits"*, *"frontal, museum-like presentation with centered, symmetrical compositions"*.

This is a second mode collapse, downstream of ours, contributed by the image model and by the fact that much of what varies in the text ("post-modern", "appropriated source") lands in the same visual place.

![Figure 6. Text versus vision diversity on 97 rendered artworks. (c) Pairwise similarities in the two spaces correlate at only r = 0.170.](figures/fig12_vision.png)

*Figure 6. Text versus vision diversity on 97 rendered artworks. (c) Pairwise similarities in the two spaces correlate at only r = 0.170.*

![Figure 7. Sixteen renders from each policy, same generator and budget. Top row: naive prompting, a corpus with a 0.000 exact-duplicate rate and healthy lexical diversity that still collapses to one visual mode — magenta and cyan collage, classical bust, barcode, neon lettering. Bottom row: max-min steering with literal-space repulsion, where palette, medium, composition and register all vary. The bottom row remains fonder of grids and tiled panels than a human art director would be; §I.6.6 says why the objective does nothing to discourage that.](figures/fig13_contact_sheet.png)

*Figure 7. Sixteen renders from each policy, same generator and budget. Top row: naive prompting, a corpus with a 0.000 exact-duplicate rate and healthy lexical diversity that still collapses to one visual mode — magenta and cyan collage, classical bust, barcode, neon lettering. Bottom row: max-min steering with literal-space repulsion, where palette, medium, composition and register all vary. The bottom row remains fonder of grids and tiled panels than a human art director would be; §I.6.6 says why the objective does nothing to discourage that.*


A reader looking at the bottom row will notice that it still favours grids, panels and tiled compositions more than a human art director would. That is what the objective asks for and no more. A max-min corpus is scored on how far apart its items are, and nothing in the score says the corpus should look like any particular distribution of artworks — there is no likelihood term, no reference set, no penalty for sitting in a region of image space that real post-modern art rarely occupies. If the generator's prior happens to place a whole family of mutually distant images inside the tiled-grid region, spreading points is satisfied by staying there and varying what fills the cells. Part II's objective is the one that has a reference distribution in it, and a corpus scored on coverage of human-written material inherits a pull toward where that material actually sits. Buying both at once means carrying both terms, which is a different optimization from the one measured here.

The vision-steered arm closes the loop where the product actually lives. It renders a bounded sample of accepted instructions, embeds them with CLIP, and feeds two things back into the text-side loop: a least-squares map from the crowded *image* directions into instruction-embedding space, so the text-side orthogonality term can push away from visual redundancy it cannot itself perceive; and mined *visual* attractors from the vision judge, appended to the same append-only ledger as the textual ones and repelled against in subsequent prompts. It is the paper's mechanism applied one level down: the ledger already repels against what the model keeps saying, and now also against what it keeps showing. It is the best arm in the table.

#### I.6.7 Attractor mining works, and is legible

The mining step produces findings specific enough to act on. From the poetry pilot, round 1 (*n* = 25): *"self-correction and immediate retraction: speakers repeatedly interrupt their own claims"*; *"ceremonial or institutional address: the voice of a bell-ringer, town crier, registrar, witness"*; *"threshold imagery and delayed passage: doors, gates, windows, bridges, shores"*. By round 2 (*n* = 50) it tracks the corpus's *drift* rather than restating round 1, noting that technical vocabulary is now being placed inside mythic frames and that recursive epistemic backtracking has become structural.

These findings are nameable and therefore repellable, which is what makes them usable as prompt constraints; a finding of "similar tone" would not be. And several of them are artifacts of *our own axis elicitation* — asking for craft-level axes like "relationship to its own claim" reliably produces self-correcting speakers. **The system's own conditioning becomes the next attractor.** The ledger catches the system's own habits alongside the model's.

#### I.6.8 The exam bank: a judged enemy-item radius

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
#### I.6.9 Head-to-head against a human-written exam bank

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

That gap is the honest measure of what is left. MMLU's spread comes from 57 genuinely different subjects; ours comes from an axis lattice a single model proposed in one call, refined a handful of times. The lexical result says our surface variety already exceeds human-authored items; the semantic result says our *conceptual* variety does not, and that closing the remaining 31% is a question about how much genuinely different subject matter the generator can be induced to reach, which is precisely the reachable-dimension question of §I.3, not a tuning problem.

#### I.6.10 Which metrics are independent of the objective

Our selection rule maximizes a weighted sum of embedding-space orthogonality and embedding-space min-gap. We then report embedding-space diversity metrics. Those two facts are not independent: the centered Vendi Score is a monotone function of how flat the embedding Gram spectrum is, which is close to exactly what the orthogonality term climbs, and the median nearest-neighbour distance *is* the min-gap term. To that extent, the wins on centered Vendi and median nearest-neighbour distance are partly tautological: the method optimizes them and the baselines do not, so they should be read as confirmation that the optimizer works rather than as independent evidence.

The independent evidence is that the method never observes the *literal* metrics at all. It reads embeddings; it has no access to token counts, *n*-gram overlap, or string identity. So exact-duplicate rate, distinct-2, 4-gram self-repetition, and *n*-gram Vendi are independent of the objective in a way the embedding metrics are not, and the method wins those too, including against the human-written bank on two of the four. Sorted by how much weight they can bear:

- **Embedding-space measures (Vendi, NN distance):** partly circular; confirmation that the optimizer works.
- **Literal-space measures (duplication, distinct-2, self-repetition, *n*-gram Vendi):** independent, since nothing in the method targets them. These carry the argument.
- **Rendered-artifact measures (CLIP, CLAP, MERT):** the most independent, living downstream of a second generative model the method never sees, which is why §I.6.6 matters more than its length suggests.

A reader who trusts only the third category still has the *r* = 0.170 result, which is a finding about every method in the table rather than a comparison between them.

---

#### I.6.11 Literal-space repulsion: fixing what the embeddings cannot see

Embedding metrics miss an entire class of repetition: tiled grids of one cell, prominent typography, a single shared palette, the same composition recolored. Audited with deliberately dumb, non-semantic signatures (a 16×16 luminance layout map, an autocorrelation tiling score, a hue histogram): half the images had a layout twin above 0.5 cosine, 40–45% were literal tilings, and 63–78% shared a palette, while prompt-level Jaccard sat at a healthy 0.14–0.17. Varied words, one visual mode: the conditional-dimension gap operating inside the *renderer*.

The fix is a four-quadrant repulsion ({text, vision} × {literal, latent}) where the two literal quadrants were previously unpopulated: content-word overlap penalties and live overused-word bans on the text side; and on the vision side, measured structural bans injected into prompts (grids banned when recent renders tile, dominant hue pairs named and banned, layout-change demands when layouts collide), plus a learned text→bad-structure bridge that penalizes candidates near prompts whose renders tiled. Rerun at matched budget, palette twins halved (0.78 → 0.35 in the coverage arm), the coverage objective improved 21% (0.206 → 0.247), and layout/tiling moved modestly (0.51 → 0.46–0.48; 0.45 → 0.33) — a partial improvement, with the residue attributable to the image model's own prior resisting text-side instruction.

#### I.6.12 Audio: prompts, the steered corpus, and embedder dependence

**Prompt-level diversity.** At matched *n*, matched prompt length and the same generator, the conditioned arm reaches centered Vendi 43.03 against naive's 22.99 (1.87×), halves 4-gram self-repetition (0.140 vs 0.291), raises distinct-2 (0.545 vs 0.348) and *n*-gram Vendi (75.3 vs 56.0), and holds roughly three times the room between nearest neighbours (0.079 vs 0.027), on 100 prompts per arm. Neither arm produces exact duplicates, so the effect is semantic. This replicates on a third artifact type the pattern of §I.6.3 and §I.6.5: conditioning moves the number, sampling temperature does not.

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

#### I.6.13 Scale and cost

The text corpora comprise 43,171 real generations for roughly $8.50 of OpenRouter spend, plus 690 rendered images across seven arms at two quality tiers, 100 rendered Lyria instrumentals, and 236 adjudicated exam-item pairs. Both `naive` arms reach *n* = 10,000; the reimplemented baselines reach 2,500 each. Every comparison is reported at a matched *n* that all compared arms actually reached.
---

### I.7 Relation to the coverage problem

Two objectives share the generation budget. **Max–min** is a packing objective: *k*-center-like, driven by the worst-case nearest pair, with no approximation guarantee under greedy selection. **Coverage** (given a finite budget, maximize the volume of a union of ε-balls) is a covering objective, closer to facility location, and monotone submodular, so greedy selection carries a (1 − 1/*e*) guarantee. The sections above develop the max–min side; Part II develops the coverage side across two further domains.

Each objective's strength is the other's failure mode. Packing spreads points toward the boundary and over-invests in outliers, which is why max–min is sensitive to outliers: outliers are what it is designed to seek. Covering weights regions by measure and fills the bulk, which is why it leaves the frontier sparse. Neither reduces to the other, and the tools that serve one degrade the other.

Both problems share this paper's structural constraints, and we expect the conditional-dimension results to be *more* consequential for coverage than for packing: covered volume is governed by reachable dimension, not by how many distinct prompts one can write, so the gap between how many distinct specs one can write and how many independent directions they actually move in bites harder there. The no-inverse-oracle assumption also degrades coverage's greedy guarantee, since the guarantee is relative to the best subset *of what was proposed*, and the proposal distribution is confined to a slice.

---

### I.8 Limitations

**One embedding oracle.** All geometry is `nomic-embed-text` geometry in the live runs. Whether "diverse" under one embedder transfers to another is untested here and is a real threat to any embedding-based diversity claim, including ours.

**Judge bias is unmeasured.** Language-model judges have systematic preferences, often *for* fluent and typical text, which is exactly the bias that would work against a diversity system. We use a judge for craft and validity scores and did not calibrate its error against human raters.

**Additive spec composition.** The axis scoring treats conditioning attributes as composing additively in embedding space, which is what makes a spec's position predictable before it is generated. Real interactions between prompt attributes are not additive.

**The exam-item judgments are model-made.** The enemy-item radius is calibrated against a blind adjudication by a language model under a psychometric rubric (§I.6.8), not by a credentialed psychometrician, and δ is specific to this embedder and this item type.

**Quality is scalar.** Craft is not one number, and collapsing it to one lets a system trade away dimensions of quality the judge does not score.

---

### I.9 Conclusion

The infinite-horizon diversity objective is easy to write, cheap to maintain, and largely beside the point. What determines whether a corpus can keep growing without collapsing into paraphrase is the dimension of the generator's conditional support relative to the manifold it lives in, and every practically important consequence follows from that one gap: saturation arrives at rate $n^{-1/m}$ rather than $n^{-1/d}$; a single prompt covers an $ε^{d-m}$ fraction of what the model could write; and only transverse prompt motion raises the ceiling.

Given an embedding oracle and no inverse, the system cannot compute its way to the next item. It can only choose what to condition on. The axis-scoring rule (spread, transversality, independence, headroom) is our answer to that choice, and its most valuable output is not the ranking but the **refine** signal: the moment it reports that nothing available is transverse any more is the moment the horizon has actually been reached, and the only remaining move is to ask the generator to subdivide its own vocabulary of variation.

That recursion is the single mechanism we found that changes the asymptote rather than the constant. Everything else (better selection, more candidates, higher temperature) buys a constant factor against a problem that is asymptotic, and the loudest of them buys it by quietly breaking the generator.

The measurements make the stakes concrete in a way the theory could not. A strong model, asked a perfectly reasonable question ten thousand times, returns the same item 1,637 times; the temperature knob barely moves that number and conditioning nearly erases it. Two diversity metrics computed on the same growing corpus point in opposite directions. And the text embeddings every method in this literature optimizes turn out to explain about two percent of the variance in whether the rendered images look alike. Each of those is a reason to distrust a single diversity number, and together they are the argument for the practice we ended up recommending: measure at the level of the artifact you are shipping, report the literal and the latent separately, publish the similarity distribution your kernel is operating on, and count your duplicates before you compute anything else.

---


---

# Part II — Coverage: reach as much as possible


### Abstract

Part I asks how to generate an unbounded stream of items that never collapses
onto its own past, under a packing objective: keep every new item far from
everything already made. This part studies the covering objective. A fixed budget of n = 10,000 items
must be spent so that the resulting set *covers* as much of the generator's
reachable behavior space as possible: we maximize the measure of the union of
small ε-balls centered at the chosen embeddings, estimated by Monte Carlo
against a reference pool of cheap draws from the generator itself. The
coverage functional is monotone submodular, so greedy marginal-gain selection
over a candidate stream inherits the classical (1 − 1/e) guarantee, and its
Monte-Carlo estimate concentrates at the Hoeffding rate; we verify all three
facts numerically rather than taking them on faith (0 violations of
diminishing returns in 5,000 randomized chain trials; greedy attains the
exact optimum on all 10 enumerable instances; the empirical 95% error of the
pool estimate sits inside the Hoeffding band at every pool size tested). We also measure a *reachability gap* — the 17.4
points of coverage that no selection rule can recover because the proposal
distribution cannot reach them, 92.5% of which recursive refinement does
recover, which follows from having an embedding oracle but no inverse. The
estimator is validated against analytically known answers, and evaluation is
budget-matched on real gpt-5.6-luna corpora with a stratum-balanced reference
and a scoring half held out from the selector. Coverage-greedy selection beats
all five literature baselines -- SemDeDup, farthest-point/k-center,
Self-Instruct's ROUGE-L filter, greedy MAP-DPP, and uniform random -- by
1.043x on DALL-E instructions and 1.091x on psychometric items.
Against the released corpora (Alpaca, PersonaHub, WizardLM), covered
fraction at a fixed radius is not comparable across corpora with different
intrinsic scales -- it correlates -0.991 with within-corpus spacing and -0.986
with Vendi -- so we use a scale-free estimator with reference-side k-NN radii.
On it, coverage-retrieval conditioning places first of twelve corpora at
0.4441 against Alpaca's 0.3722, at one-twentieth of Alpaca's generation
budget and with the highest precision in the field (0.973). The same data
shows the covering/packing dissociation at full strength: k-center maximizes
min-gap (0.696), and finishes last on coverage (0.096, a fifth of random), and
the two highest-Vendi selectors are the two worst covering ones. Naive
psychometric prompting is 73.6% exact duplicates (one item repeated 2,726
times in 10,000 generations), which high temperature barely dents (71.0%) and
conditioning nearly eliminates (0.0-4.7%) -- direct evidence that
conditioning, not temperature, buys coverage, and a demonstration that
nearest-neighbor epsilon-calibration is undefined on such a corpus (median
self-distance exactly 0). Finally, literal and latent diversity
move in opposite directions as the corpus grows (distinct-2 falls
0.944 → 0.770 while mean-centered Vendi rises 3.98 → 37.06), and the
uncentered Vendi is compressed 2.7× by the embeddings' shared mean
direction — so both levels, and both centerings, need reporting.

### II.1 Introduction

A team building an evaluation suite for a language model does not get an
infinite horizon. They get a budget (ten thousand prompts, say), and a
product requirement that is naturally a *coverage* requirement: the suite
should contain an example near every qualitatively distinct behavior the
system under test might encounter, because a suite that clusters into a few
families gives false assurance about everything outside them. The same shape
of problem appears when generating synthetic user personas to exercise a
dialogue system: near-duplicate personas waste budget, and bugs live in the
parts of user-behavior space the corpus never visits.

The parent paper in `research/infinite_horizon_diversity/` (this work lives
in its `coverage/` subdirectory and imports its modules directly) studies the
open-ended version of this problem: an unbounded stream, an objective that
combines typicality (mean distance to prior embeddings) with max-min novelty,
and a method stack — generator-elicited latent axes, spectral
orthogonalization against the corpus second-moment matrix, recursive
refinement of saturated cells, and an append-only attractor ledger — that
keeps the stream from collapsing at any horizon. This paper keeps the method
stack and swaps the objective, because the budgeted problem is *not* the
truncation of the infinite one:

1. **Max-min packing is the wrong objective at a finite budget.** Packing
   pushes points apart; its optima sit on the boundary of the reachable
   region and over-invest in extremes and outliers, because an outlier is by
   construction far from everything. Coverage weights regions by how much
   probability mass they hold and fills the bulk first. The two objectives
   correspond to two different classical problems — k-center versus maximum
   coverage / facility location, and their optima genuinely differ (§II.3.4,
   §II.7.2).
2. **A known n permits planning.** With an unbounded horizon there is nothing
   to allocate; with n = 10,000 known in advance one can ask how the budget
   should be split across cells of the behavior space — equally,
   proportionally to measure, or adaptively, and the answer depends on the
   radius ε at which coverage is demanded (§II.3.5, §II.7).
3. **Coverage needs a denominator.** "We covered 62% " is meaningless without
   saying 62% *of what*. In the pipeline the
   denominator is the generator's own reachable distribution, approximated by
   a pool of cheap draws. For comparing corpora built by different methods it
   is a held-out human-written reference under a scale-free estimator whose
   radii are a property of the reference alone — fixed, identical for every
   corpus scored, and impossible for the method being graded to grow (§II.5).

Everything is built to the same scaling discipline as Part I:
per-item cost is O(1) in n (fixed-size reference pool, incremental covered
masks, bounded history subsamples), so nothing in the pipeline gets slower at
item 10,000 than at item 100.

We make four contributions. First, a head-to-head against released
instruction corpora — Alpaca, WizardLM Evol-Instruct, Persona-Hub — scored by
a scale-free coverage estimator against a held-out human-written reference no
corpus was aimed at, which our method wins at matched evaluated *n* on a
twentieth of the generation budget (§II.5). Second, a coverage adaptation of
Part I's axis-scoring rule in which the four-factor product collapses to a
single estimable scalar (the expected marginal ε-ball gain of conditioning on
the axis) with an ablation isolating what score-guided conditioning buys
(§II.4, §II.5). Third, a live pilot of the full method (elicited axes, score-guided
focus, judge gate, ledger, refinement, coverage-greedy selection) on
gpt-5.6-luna with local embeddings (§II.6). Fourth, a formalization of budgeted
corpus construction as Monte-Carlo maximization of a union-of-balls coverage
functional, with the relevant guarantees stated precisely and *checked
numerically* (§II.3, §II.7), including the no-inverse-oracle propositions and a
direct measurement of the reachability gap they imply (§II.2, §II.7.6).

### II.2 Problem statement

Fix an embedding map φ into R^D (we use unit-normalized 768-d embeddings), and a radius ε > 0. The generator, prompted in
whatever ways the pipeline can reach, induces a *reachable distribution* μ
over R^D: the pushforward of everything the generator can actually be made to
produce. Given a budget n, choose a set S = {x_1, …, x_n} of generated items
to maximize the covered measure

  F_μ,ε(S) = μ( ⋃_{x∈S} B(x, ε) ),

the probability that a fresh draw from the generator lands within ε of some
chosen item. Equivalently, S should be an ε-net for as much of μ as the
budget allows. Three deliberate choices:

- **Coverage of the reachable space, not of R^D.** Balls in empty space cover
  Lebesgue volume but no behavior. Measuring against μ makes the metric mean
  "fraction of what the generator can do that the corpus has an exemplar
  of", which is the product requirement for an eval suite. The denominator is
  therefore stated next to every coverage number; where corpora built by
  different methods are compared, it is a held-out human-written reference
  identical for all of them (§II.5).
- **ε is a resolution parameter, not a nuisance.** Small ε asks for exemplars
  of fine behavioral distinctions; large ε only for one exemplar per coarse
  region. Every result below names the radius, or the range of radii, it is
  reported at, because the ranking of methods changes with ε.
- **Quality is a constraint, not part of the objective.** An item that covers
  new territory but fails a quality/typicality bar is not an asset, so the
  judge gate rejects it regardless of the gain it would have scored (§II.4).

**The oracle asymmetry.** One assumption structures everything downstream: we
have an embedding oracle E: text → R^D, and *no inverse*. There is no E⁻¹
that decodes a target point back into text. We can compute exactly where in
embedding space the next item should land — the largest uncovered region, the
trailing eigenspace, but we cannot manufacture an item that lands there. The
only handles a pipeline has are (a) language-valued latent conditioning that
*steers* the generator's proposal distribution, and (b) selection among
sampled candidates. This is why the architecture is propose–measure–select
rather than solve, and it has two consequences we state as propositions:

> **Proposition 1 (no direct construction).** Coverage-optimal point sets are
> not directly constructible. The greedy (1 − 1/e) guarantee (§II.3.2) holds
> relative to the best subset of the *proposal distribution's support*, not
> of R^D; the difference between what an unconstrained selector could cover
> and what a proposal-limited selector can is a *reachability gap*, which is
> a property of the generator-plus-conditioning stack, not of the selection
> algorithm. We measure it directly in §II.7.6.

> **Proposition 2 (conditioning as surrogate inverse).** Language-valued
> conditioning is the only available surrogate for E⁻¹: choosing which axis
> to condition on next is choosing which *slice* of the reachable manifold
> the next candidates will be sampled from, so a axis scoring that ranks axes by
> where their slices land (§II.4) is precisely an approximate inverse — it maps
> "where we want mass" to "which words to condition on".

### II.3 Theory

#### II.3.1 The coverage functional is monotone submodular

For any measure μ and radius ε, define F(S) = μ(⋃_{x∈S} B(x, ε)) on finite
S ⊂ R^D. F is monotone (adding a ball cannot uncover anything) and
submodular: for S ⊆ T and any x,

  F(S ∪ {x}) − F(S) ≥ F(T ∪ {x}) − F(T).

*Proof sketch.* The marginal gain of x given S is μ(B(x, ε) \ ⋃_{y∈S}
B(y, ε)). Since S ⊆ T implies ⋃_S B(y,ε) ⊆ ⋃_T B(y,ε), the set being
subtracted only grows, so the gain only shrinks. This is the standard
coverage-function argument; nothing about balls is essential — any map from
items to measurable "footprints" yields a submodular F. ∎

The same argument applies verbatim to the Monte-Carlo estimator we actually
optimize. Given a reference pool P = {p_1, …, p_P} of iid draws from μ,

  F̂(S) = (1/P) · #{ p ∈ P : dist(p, S) ≤ ε }

is itself a finite coverage function (item x covers the fixed subset
N_ε(x) = {p : ‖p − x‖ ≤ ε} of the pool, and F̂(S) = |⋃_{x∈S} N_ε(x)|/P) so
F̂ is monotone submodular exactly, at every sample of the pool. Our
implementation check (§II.6.1) found 0 violations of either property in 5,000
randomized nested-chain trials, as it must if the code is right.

#### II.3.2 Greedy and its guarantee

Because F̂ is monotone submodular with F̂(∅) = 0, the greedy algorithm that
repeatedly adds the item of maximum marginal gain satisfies the
Nemhauser–Wolsey–Fisher bound

  F̂(S_greedy) ≥ (1 − 1/e) · max_{|S|≤n} F̂(S) ≈ 0.632 · OPT,

and this is tight for the class (maximizing coverage is NP-hard, and beating
1 − 1/e is hard under standard assumptions, Feige 1998). §II.7.3 checks the
bound against *exact* optima on instances small enough to enumerate; greedy
attained the optimum itself on all ten instances, comfortably above the
bound — typical behavior, since the 1 − 1/e worst case requires adversarial
structure.

#### II.3.3 Sequential generation is streaming greedy

A generation pipeline cannot pick from a precomputed ground set: candidates
arrive a few at a time (K per step), and must be accepted or discarded
immediately. This is the streaming submodular maximization regime.
Sieve-streaming (Badanidiyuru, Mirzasoleiman, Karbasi, Krause 2014)
maintains geometric thresholds and accepts any arrival whose marginal gain
clears one, achieving a (1/2 − δ) guarantee in one pass with O((k log k)/δ)
memory, independent of stream length. Our per-step rule (take the best of K
fresh candidates by marginal gain) is the natural best-of-K relaxation:
weaker in the worst case (an adversarial stream can starve it) but
stronger on exchangeable streams like ours, where each batch is iid from the
current proposal distribution and thresholding would only discard budget.
Empirically the K = 4 stream rule retains most of full greedy's value over an
identical ground set (0.484 vs 0.528 covered fraction; §II.6.2) at a per-item
cost that never touches the whole ground set. What matters for scaling is
that the marginal-gain oracle is O(K · P) per step against a fixed-size pool
with an incrementally maintained covered mask — O(1) in n.

#### II.3.4 Coverage is not packing

Max-min selection (choose the candidate farthest from the accepted set) is
the greedy algorithm for the k-center / packing objective, and Part I
paper shows it is the right *shape* of objective for an unbounded stream once
a typicality anchor keeps it on-manifold. At a finite budget against a fixed
μ the two objectives pull apart:

- k-center cares about the worst-covered *point of S itself*; its optima
  push to the boundary of the support and place points in regions of
  vanishing measure (an outlier is far from everything, so packing *rewards*
  it), and under a fixed budget every such item is a ball spent covering
  ≈ 0 measure.
- Maximum coverage / facility-location-type objectives weight territory by
  measure: a second exemplar in a heavy mode can be worth more than a first
  exemplar of a negligible one, and junk is worth nothing because μ puts
  almost nothing near it.

§II.6.2 isolates the contrast with no generator in the loop: on one shared
candidate set, greedy-coverage covered 0.528 of a held-out pool where
farthest-point packing covered 0.143, while packing's min-gap (1.44) beat
greedy's (0.577) by 2.5×. Neither is "better"; they optimize different
functionals, and §II.7.2 shows the same double dissociation with the full
generation loop in place.

#### II.3.5 Estimation error of the Monte-Carlo coverage

For fixed S, each pool point contributes an iid Bernoulli(F(S)) indicator,
so Hoeffding/Chernoff gives

  P( |F̂(S) − F(S)| > t ) ≤ 2 exp(−2 P t²),

i.e. a 95% band of t₉₅ = sqrt(ln(2/0.05)/(2P)) — about ±0.030 at P = 2,000
and ±0.015 at P = 8,000. Two caveats we take seriously. First, the bound is
for fixed S; a set *selected* by optimizing F̂ on the same pool is biased
upward on that pool (the winner's curse), which is why every
coverage number reported here is computed on held-out material the selector never saw, and the
live pilot reports selection-pool coverage explicitly labeled as such.
Second, uniform-over-S guarantees would need a union bound over the
(exponentially many) candidate sets; we do not rely on one — held-out
evaluation makes the fixed-S bound the relevant one. §II.7.4 confirms the band
empirically: at every pool size tested, ≥ 99% of 200 independent pool
estimates fell within t₉₅ of a 200k-pool ground truth.

#### II.3.6 Allocation under a known budget

Suppose the space decomposes into cells (modes) with measures w_m, and
within-cell placement is near-saturating: with c_m items in cell m, the
covered measure of that cell is approximately w_m · g_m(c_m) with g_m
increasing and concave (each additional exemplar covers a shrinking uncovered
remainder — submodularity again, now per cell). Maximizing Σ w_m g_m(c_m)
subject to Σ c_m = n is a concave allocation problem whose optimum is the
water-filling rule: equalize marginal gains w_m g_m′(c_m) across cells. Three
closed-form rules bracket the regimes:

- **ε large relative to cell diameter** (g_m saturates after ~1 item): any
  rule that puts at least one item per cell is near-optimal; equal
  allocation wastes least; proportional over-invests in heavy cells.
- **ε small** (g_m near-linear over feasible c_m, slope ∝ per-item covered
  mass ≈ density × ball volume): marginal gain is w_m-weighted throughout,
  and proportional-to-measure allocation dominates equal.
- **In between**, neither closed form matches water-filling; greedy marginal
  coverage *is* water-filling implemented empirically, and wins at the
  radius it optimizes.

The measured allocation experiment (§II.7.5) shows exactly this pattern, plus a
failure mode the clean theory hides: once greedy has saturated the pool at
its selection radius, its marginal signal is identically zero and the
tie-breaking rule silently decides where the rest of the budget goes.

**Depth per condition and switch cost.** Part I's slice theory sharpens the
within-cell side of allocation. For a *fixed* prompt/spec the generator concentrates on an
m-dimensional slice with m ≪ d, so fixed-prompt novelty decays as n^(−1/m)
(fitted slopes −0.472, −0.309, −0.195 for m = 2, 3, 5 — governed by m, not by
the ambient dimension), and one slice ε-covers a vanishing ε^(d−m) fraction
of the manifold. The budgeted consequence is a breadth-vs-depth rule whose
correct form (their P3, corrected from its first statement) is: with *free*
condition-switching the optimum is the boundary n\* = 1 — one sample per
spec, then move, and an interior optimal depth exists only once switching
carries a real cost c (eliciting a spec, embedding levels, an occasional
refinement call): their measured n\* rises from 1 (c = 0) to 10 (c = 3) to
30 (c = 30). Optimal depth is set by the switch-cost-to-sample-cost ratio,
not by ε alone. Our pipelines sit near the cheap-switching regime — in the live pilot we
measure c ≈ 0.10 generation-equivalents (§II.6: 5 axis-elicitation, refinement and
ledger-mining calls amortized over 50 steps, against 3 generations per
step), which is why every policy here draws its K candidates from K *fresh*
specs rather than sampling any spec deeply. At that switch cost Part I's
sweep puts the optimum at or adjacent to n\* = 1, which is what we do.

### II.4 Method

The pipeline is Part I's **Recursive Axis Conditioning** (RAC) stack with the selection
objective and its bookkeeping swapped from packing to covering; we write RAC-coverage for
this instantiation and RAC-packing for Part I's. Concretely:

**Reachable pool (the denominator).** Before selection begins, draw a pool of
cheap, unconditioned samples from the generator and embed them. This pool
*is* the Monte-Carlo measure: coverage means coverage of it. Live, the pool is 240 one-shot prompts. Cost
is O(P) once, amortized over the whole run.

**Elicited axes and specs.** The generator is asked (before generating any
items) to name the latent axes along which items in this domain can differ,
each with discrete language-valued levels (live: 6 axes for evaluation
prompts, e.g. persona pressure, trap construction, response obligation). A
spec is one level per applicable axis; K specs are sampled per step and one
candidate is generated under each. Axes give the proposal distribution
spread the base generator lacks; they are the reason the method's candidates
are worth ranking at all.

**Score-guided conditioning (the coverage adaptation).** Part I's generation
axis scoring ranks candidate axes by a four-factor product — spread (are the axis's levels
different from each other), transversality (does their variation point
outside the corpus's occupied eigenspace), independence (measured against
axes already in use, never against rival candidates — scoring candidates
against each other makes duplicates zero each other out), and headroom (an
entropy deficit over the axis's sampled levels). Under a *coverage* objective
this product collapses, and the collapse is itself a packing-vs-covering
result. Packing has no per-axis marginal decomposition (min-gap is a global
property), so the axis scoring must approximate "will this axis move the slice
somewhere new" from four separate geometric proxies. Submodular coverage has
an exact marginal quantity: the **expected marginal ε-ball gain of
conditioning on the axis**, E_{x∼axis}[F(S ∪ {x}) − F(S)]. That one scalar
subsumes the four factors: an axis with near-synonym levels (no spread) or
whose slices sit inside covered territory (no transversality) or which
duplicates an axis already exploited (no independence, because the covered
mask already contains that axis's contribution and re-scores every candidate
after every accept — the same greedy re-scoring the axis scoring performs
by hand, executed here by the objective itself) or whose region is
already covered (no headroom) all have small expected marginal gain, and the
headroom that remains is *measure-weighted* by construction — a large
under-sampled region beats a small one because it holds more uncovered pool
mass. We keep Part I's embedding-based spread and
transversality (level-description embeddings against the accepted corpus's
occupied eigenspace), and replace its entropy headroom with the
measure-weighted estimate (per-level mean observed marginal gain, optimistic
for unused levels); the top-two axes by this promise become the step's
dominant design pressures, with their most-different levels chosen by
farthest-point selection in level-embedding space — the axis scoring acting as
the surrogate inverse of Proposition 2.

**Coverage-greedy selection.** Each step: embed the K candidates, compute
each one's marginal gain — the number of not-yet-covered pool points within
ε_sel, and accept the gated candidate of maximum gain (ties broken by judge
score). The covered mask is updated incrementally; per-step cost is
O(K · P + P) regardless of n.

**Judge gate (quality/typicality).** A separate LLM call (generation never
self-grades) scores each candidate on a rubric; candidates below the bar are
ineligible regardless of gain. The gate is negative supervision only: if every candidate
fails, the step falls back to best-judged rather than emitting nothing, and
the rejection is logged.

**Attractor ledger.** Every 20 accepted items (50 in Part I), the accepted corpus is shown to the model with the question "what do
these have in common?"; the mined attractors are appended to a JSONL ledger,
and a bounded top-K slice becomes explicit negative constraints in subsequent
generation prompts. The ledger is append-only; its influence on any single
prompt is fixed-size.

**Recursive refinement + pool refresh.** When recent accepted items' marginal
gains saturate (the region's balls mostly re-cover covered ground), the
responsible region of spec-space is described back to the generator, which
either splits an axis level into finer sub-levels or mints a new axis
that applies inside that cell — conditional axes forming a tree, exactly as in
Part I. The budgeted setting adds one obligation the
infinite-horizon setting does not have: refinement changes the reachable
distribution, so the reference pool must be refreshed with draws from the
newly opened region, or the selector will see zero gain precisely where the
new territory is: its balls cover no *old* pool points.

### II.5 Comparison with released instruction corpora

We compare against the released corpora of the three most-used synthetic-instruction methods — Alpaca (Self-Instruct, 52k), PersonaHub (50k), and WizardLM Evol-Instruct (143k) — on scale-free coverage of human-written instructions, together with a budget-matched ablation of our own configurations.

#### II.5.1 Protocol

Human-written reference: databricks-dolly-15k, split into two disjoint 3,000-item halves — a STEER half that our method may read as embeddings on the selection side, and an evaluation half that nothing in any pipeline ever reads, and at which no corpus (ours or released) was ever aimed. All scores below are on the evaluation half. Estimator: Naeem et al. (2020) coverage/density with reference-side k-NN radii, reported as AUC over k ∈ {3, 5, 10, 20}; radii are a property of the reference alone, identical for every corpus, with nothing tunable per corpus. Every corpus is evaluated as a uniform random sample at matched *n* = 450. Our arms and the reimplemented baselines all receive the same 175 human seed tasks (the seed set Alpaca was built from), the same generator, and the same budget of 2,400 generator calls — rejection and selection losses are counted, not hidden. The generator never sees a reference word in any arm.

#### II.5.2 Method

1. **Aim by retrieval.** Sample an uncovered reference region (density-weighted, ∝ 1/r²), and retrieve the nearest texts we already own (seed tasks plus our own corpus) as the few-shot exemplars. The reference enters as embeddings on the steering side only; retrieval over owned text substitutes for the missing inverse oracle.
2. **Radius-adaptive mode.** The target's own k-NN radius selects the prompt mode: tight radius (dense region) -> imitate the local task family closely; wide radius (sparse region) -> full diversification, with the nearest own outputs shown as explicit in-prompt negatives.
3. **Literal-channel spread.** Level-usage balancing across the elicited axis lattice (the headroom term applied at conditioning time), and a terse-register mandate.
4. **Keep everything.** Coverage is monotone in items; at a generation-matched budget, every discard is a permanent loss.

#### II.5.3 Results

![Scale-free coverage of a held-out human-written reference. (a) Coverage at every reference-side radius. (b) All twelve corpora at matched evaluated n = 450. (c) Coverage against precision: the winning corpus is also the one that stays inside the reference manifold.](figures/fig_h2h_scalefree.png)

*Scale-free coverage of a held-out human-written reference. (a) Coverage at every reference-side radius. (b) All twelve corpora at matched evaluated n = 450. (c) Coverage against precision: the winning corpus is also the one that stays inside the reference manifold.*


All twelve corpora, evaluation half, matched *n* = 450:

| rank | corpus | AUC | precision |
|---|---|---|---|
| 1 | **RAC-coverage, retrieval-aimed (2.4k)** | **0.4441** | **0.973** |
| 2 | Alpaca (Self-Instruct, 52k) | 0.3722 | 0.87 |
| 3 | RAC-coverage, selective 1-of-8 (304) | 0.2591 | 0.93 |
| 4 | PersonaHub (50k) | 0.2532 | 0.78 |
| 5 | WizardLM Evol-Instruct (143k) | 0.2329 | 0.77 |
| 6 | Self-Instruct (reimpl., same seeds/budget) | 0.2188 | |
| 7 | RAC, conditioning keep-all | 0.2127 | |
| 8 | few-shot from seeds | 0.2114 | |
| 9 | RAC, orthogonalized conditioning | 0.2102 | |
| 10 | axis-conditioned, unseeded | 0.1008 | |
| 11 | naive | 0.1003 | |
| 12 | high temperature | 0.0947 | |

Our method places first, 19% above Alpaca, with the highest precision in the field, at roughly one-twentieth of Alpaca's generation budget. An interim measurement at 22% of budget (n = 528) already scored 0.4276 against Alpaca's 0.3666 on the same protocol, so the result does not depend on final corpus size. PersonaHub and WizardLM score below our selective arm despite 20–60× the scale.

Budget-matched ablation (full corpora, 2,400 generator calls each, same evaluation half):

| configuration | kept | AUC | density | precision | recall |
|---|---|---|---|---|---|
| **RAC-coverage, retrieval-aimed** | 2,398 | **0.6597** | 1.637 | **0.960** | **0.178** |
| Self-Instruct (reimpl.) | 1,833 | 0.3262 | 1.050 | 0.949 | 0.161 |
| few-shot from seeds | 2,399 | 0.3246 | 1.052 | 0.935 | 0.152 |
| conditioning, keep-all | 1,310 | 0.3147 | 0.704 | 0.864 | 0.068 |
| orthogonalized conditioning | 1,152 | 0.2941 | 0.718 | 0.873 | 0.101 |
| selective (1-of-8) | 304 | 0.2591 | 1.009 | 0.928 | 0.099 |

The ablation attributes the margin: retrieval-aiming and the radius-adaptive mode carry it (rows 1 vs 2–4), few-shot anchoring to the human seeds accounts for most of the baselines' scores (row 3 vs the unseeded arms at 0.09–0.10), Self-Instruct's ROUGE filter adds 0.0016 over bare few-shot, and both selection (row 6), and orthogonalized conditioning (row 5) reduce coverage relative to keeping everything — consistent with Part III: coverage is monotone and measure-seeking, so discarding and occupied-span avoidance are each counter-productive under this objective, whereas both are load-bearing under max-min.

#### II.5.4 Notes

**Reference-sample asymmetry.** Our method consumes a sample of the target distribution, as embeddings, on the steering side; the released corpora had no such input. This is the method's designed capability (coverage is always coverage *of* something, and a method that may specify the something should), but the like-for-like no-reference comparison is rows 2–5 of the ablation, where our no-STEER configurations sit at parity with the Self-Instruct family. The precise claim: given a specification of the space to cover, even one the generator never reads, retrieval-aimed density-adaptive conditioning covers it substantially better than the strongest seeded baseline covers it at twenty times the budget.

Coverage and internal diversity are different objectives. The winning corpus has the lowest mean-centered Vendi of its cohort (62.1; density 1.69): it spends items where the reference measure is, near-duplicates included, exactly as facility location prescribes. Ranking these corpora by a single internal-diversity score would rank the coverage winner last.

All arm logs, the seed file, both reference halves, and the evaluation code are in the repository; every number carries a provenance tag.

### II.6 Live pilot

![Live pilot: coverage and quality against budget.](figures/fig_live_pilot.png)

*Live pilot: coverage and quality against budget.*


#### II.6.1 Setup and what it cost

The pilot builds a red-team *evaluation* prompt suite on the real stack:
gpt-5.6-luna via OpenRouter for generation, judging, attractor mining and
axis elicitation; local `nomic-embed-text` (768-d, unit-normalized) for
embeddings. Severity is deliberately mild — the items are benign user
messages that probe whether an assistant adds disclaimers, resists
sycophancy and leading questions, and stays calibrated, and the judge scores
a `benign` dimension that hard-gates acceptance. All state is append-only and
checkpointed, so a run is resumable after an interruption.

The run reached its full target: **50/50 accepted items** in 50 steps
(K = 3 candidates each), against a **50-item naive baseline** and a
**360-item reference pool**. The pool has two strata — 240 unconditioned
one-shot draws and 120 axis-conditioned but *unselected* draws, because a
reference pool must be drawn from the same proposal process the policy uses,
including every conditioning mechanism. A pool of unconditioned draws alone
sits outside the region axis conditioning reaches (median distance 0.88, against
a selection radius of 0.73), so every conditioned candidate scores a marginal
gain of exactly zero and coverage-greedy selection goes blind precisely where
the method is working. Six axes were elicited, two
refinements fired (8 axes final), and two ledger rounds mined attractors that
became negative constraints. Judge quality averaged 8.89/10. Cost of the
final instrumented segment: **74 calls, 36,056 prompt + 74,681 completion
tokens, $0.0968**, 394 s wall clock. Earlier segments (pool construction, the
naive baseline, and the first 32 accepts) predate the usage ledger and are
not included in that figure; total pilot spend was well under the $3 budget
but only the final segment is *measured*, so we report that number rather
than an estimate.

#### II.6.2 Literal vs latent diversity, and a kernel caveat

Tracking both n-gram and embedding diversity as the corpus grows reproduces
a divergence reported at much larger scale on this model family: the two move
in **opposite directions**.

| n | distinct-1 | distinct-2 | Vendi (centered) | Vendi (uncentered) |
|---|---|---|---|---|
| 5 | 0.614 | 0.944 | 3.98 | 3.62 |
| 10 | 0.482 | 0.861 | 8.01 | 5.04 |
| 20 | 0.367 | 0.809 | 16.16 | 7.37 |
| 30 | 0.338 | 0.791 | 23.65 | 10.00 |
| 40 | 0.312 | 0.772 | 30.55 | 11.87 |
| 50 | 0.295 | 0.770 | 37.06 | 13.54 |

Literal diversity falls monotonically (distinct-2 0.944 → 0.770), while latent
diversity rises monotonically (centered Vendi 3.98 → 37.06). Reporting either
alone supports an opposite conclusion about the same corpus. The mechanism is
plain enough — a growing corpus reuses vocabulary and sentence frames while
still spreading in meaning-space, but it means "diversity" must be qualified
by level, and a coverage objective defined on embeddings is silent about
literal repetition. (Against the naive baseline the method is slightly *less*
literally diverse at distinct-1, 0.295 vs 0.341, and slightly more diverse
latently, centered Vendi 37.06 vs 30.34, with markedly lower self-repetition,
max 3-gram Jaccard 0.036 vs 0.083.)

The uncentered Vendi is a kernel artifact. Same-domain embeddings share a
mean direction, and an uncentered linear kernel measures that shared cone as
if it were content. On this data the two readings differ by 2.7× (13.54 vs
37.06) and, worse, they order the growth curve differently in slope. We
report both everywhere and treat only *relative* comparisons at fixed n as
defined for the uncentered form.

Is ε measuring the cone or the content?. This is the question the caveat
raises for every coverage number computed with cosine distances, so we
measured it. Our pool's mean
pairwise cosine is **0.444** (p05 0.330, p95 0.591) (a wide cone, not the
~0.9 concentration seen in some same-domain embedding sets) with mean
direction norm 0.667. Our ε_sel = 0.770 corresponds to cosine similarity
0.704, which sits at the **1.8th percentile** of pairwise distances: an
ε-ball at this radius captures genuine near-duplicates, not the bulk of the
cone. So for this pilot the coverage numbers are measuring content. We record
the diagnostic because the answer is embedder- and domain-dependent: with a
tighter cone the same calibration rule would have produced an ε inside the
cone's width, and the resulting "coverage" would have been a restatement of
the embedding geometry. Any coverage result computed on cosine distances
should publish this check.

<!--FIG:fig_live_pilot.png|Figure 6. Live pilot on gpt-5.6-luna (n = 50). Left: coverage growth of the method run versus the naive baseline at three radii on the pooled denominator, where naive leads. Center: per-item marginal gains, showing the plateau from item ~35 and the recovery after the refinement at step 48. Right: the same runs split by pool stratum -- each policy covers its own proposal distribution, which is what the pooled number obscures.-->

<!--FIG:fig_literal_vs_latent.png|Figure 7. Left: literal (n-gram), and latent (embedding) diversity move in opposite directions as the corpus grows. Right: uncentered versus mean-centered Vendi on the same sets -- the shared mean direction compresses the uncentered score by roughly 2.7x, so only relative comparisons at fixed n are trustworthy.-->

### II.7 Numerical verification of the theory

Each theoretical claim above is checked against a numerical experiment designed to break it; the underlying numbers are in the accompanying data release.

**II.7.1 Submodularity (exact).** 200 candidates, 5,000-point pool, ε = 1.0.
Over 5,000 random nested chains S ⊂ T with a fresh x: 0 violations of
diminishing returns, 0 of monotonicity, worst violation 0.0.

**II.7.2 Selection rules head-to-head (no generator).** One shared ground set
of 2,000 candidates, budget k = 300, ε = 1.0, held-out 20k pool:

| rule | covered fraction (held-out) | min-gap |
|---|---|---|
| full greedy | 0.528 | 0.58 |
| stream greedy, K = 4 | 0.484 | 0.68 |
| random | 0.420 | 0.53 |
| max-min (Gonzalez) | 0.143 | 1.44 |

Packing is catastrophic *as a coverage algorithm* (below random by 3×), while
being unbeatable at its own metric — the pure-form double dissociation.
Stream greedy with K = 4 keeps 92% of full greedy's coverage.

**II.7.3 The (1 − 1/e) bound against exact optima.** Ten instances with
|ground| = 30, k = 5 (142,506 subsets enumerated each): greedy/OPT ratio was
1.0 on every instance; bound 0.632 never approached.

**II.7.4 Monte-Carlo error.** Fixed S of 300 items, ground truth from a
200,000-point pool (F = 0.4134). Over 200 independent pools per size:
empirical 95th-percentile absolute error 0.042 / 0.020 / 0.011 at
P = 500 / 2,000 / 8,000, versus Hoeffding t₉₅ = 0.061 / 0.030 / 0.015; the
fraction of estimates inside the band was 0.99–0.995 ≥ 0.95 everywhere.
P = 20,000 (our estimation pool) implies t₉₅ ≈ 0.0096: pool noise is well
below every effect size we interpret.

**II.7.5 Allocation rules.** n = 10,000 across the redteam world's M = 60 cells
with *known* measures (the oracle planning question), evaluated against a
50k full-mixture pool:

| rule | ε = 0.6 | ε = 1.0 | ε = 1.6 | ε = 2.4 |
|---|---|---|---|---|
| equal per cell | 0.046 | 0.494 | 0.9916 | 0.9921 |
| ∝ measure | **0.060** | 0.547 | 0.9917 | 0.9921 |
| ∝ sqrt(measure) | 0.055 | 0.531 | 0.9918 | 0.9921 |
| greedy marginal (at ε = 1.0) | 0.004 | **0.618** | 0.9919 | 0.9921 |

Three regimes, as §II.3.6 predicts. At large ε every rule saturates (all
≈ 0.992 — the remaining 0.8% is junk measure no allocation can reach). At the
greedy-optimized radius, greedy beats the best closed form by 7 points:
water-filling in the wild. At small ε, proportional wins among closed forms
(near-linear per-cell returns favor measure-weighting), and greedy at
ε = 1.0 is *terrible* (0.004), for an instructive reason: greedy saturated
its ε = 1.0 pool after roughly half the budget, its marginal signal went
identically to zero, and argmax tie-breaking silently dumped 5,083 of 10,000
items into one arbitrary cell (correlation of greedy's allocation with every
closed form ≤ 0.17). Lesson for practitioners: greedy's guarantee says
nothing about what it does *after* it has won; at a finite budget you must
either select at the smallest ε you care about, or hand the post-saturation
residual to an explicit rule (e.g. proportional), or lower ε adaptively as
gains vanish.

<!--FIG:fig_allocation.png|Figure 4. Left: allocation rules at four radii -- proportional wins at small radius, greedy at its own radius, everything saturates at large radius. Right: greedy's realized allocation is measure-aware but saturating, with the post-saturation tie-breaking pathology visible as a single 5,083-item spike.-->

**II.7.6 The reachability gap (Proposition 1, measured).** Identical greedy
selection (ε = 1.0, k = 300, |ground| = 2,000), three proposal
distributions, one held-out full-mixture pool. An *oracle* ground set drawn
from the full latent mixture (the best a selector could do if conditioning
could reach everything) covers 0.387. A *proposal-limited* ground set drawn
from the base specs (which reach 25 of 60 modes) covers 0.212: a
reachability gap of 17.4 points that no selection algorithm can close,
because the deficit is in what the generator can be made to emit, not in how
candidates are chosen. A *refined-proposal* ground set (one spec per mode —
the target recursive refinement works toward) covers 0.374, recovering 92.5%
of the gap. This is Proposition 1 made quantitative: the (1 − 1/e) guarantee
binds relative to the proposal support, and conditioning (not selection) is where the missing coverage lives.

**II.7.7 Lattice size is not reachable dimension.** Using Part I's
slice-structured world, in which specs compose level directions additively,
we build two worlds with *identical* lattice size
L^A = 1,024 distinct specs but different rank of the level-direction span:
full-rank (rank 10, reachable dimension 13) versus low-rank (all axes forced
into a shared 2-dim subspace: rank 2, reachable dimension 5). Same budget
(n = 2,000 random-spec draws), same radius (ε = 0.27): the low-rank world's
own reachable pool is 68.1% covered where the full-rank world's is 49.0%.
The number of distinct prompts you can write says nothing about how much
there is to cover; covered volume is governed by reachable dimension, and a
huge lattice over low-rank level directions is a thin space wearing a
combinatorial costume. For coverage this distinction is sharper than for
packing, because the denominator itself is set by reachable dimension.

**II.7.8 Refinement direction: density vs reach.** Part I's negative
result — naive refinement hurt, because tight children near saturated
parents concentrate mass where the corpus already sits — ports to coverage,
with a measurement subtlety worth recording. Nearest-neighbor distance
cannot distinguish density-refinement from reach-refinement: child draws sit
≈ 1.04 from the nearest corpus point whether they moved in-span or off-span,
because a 1,500-item corpus is sparse in its own 13-dimensional occupied span.
Density-versus-reach is a *span* property, and must be measured with span-level
metrics: *local* children (tight,
0.25× displacement — the naive move) produce draws of which 36.7% are
already covered by the pre-refinement corpus at the coverage radius, and add
+0.02 effective dimensions — almost pure density. Full-length *isotropic*
displacement produces 0% pre-covered draws and +0.13 effective dimensions;
*transversality-guided* displacement 0% and +0.14, with off-span
displacement energy 0.915 vs 0.823 unguided. The honest reading: (i) the
damning comparison is local-vs-far, and "split the saturated cell into
tighter sub-levels" is only worth its cost if the children actually move;
(ii) guidance beats unguided displacement, but by little *here*, because
this corpus occupies only a thin span (6 of 64 dimensions at the 95% energy
level), and a random direction is already 82% transverse to it — guidance
matters most when the corpus has already spread, which is exactly the
late-run regime where refinement fires.

### II.8 Related work

![Selection benchmark: coverage-greedy against five literature baselines.](figures/fig_benchmark.png)

*Selection benchmark: coverage-greedy against five literature baselines.*


**Submodular maximization.** The (1 − 1/e) greedy guarantee for monotone
submodular objectives is Nemhauser, Wolsey & Fisher (1978); hardness of
beating it for max-coverage is Feige (1998). Lazy greedy (Minoux 1978) and
stochastic greedy (Mirzasoleiman et al. 2015) make greedy cheap;
sieve-streaming (Badanidiyuru et al. 2014) gives the single-pass regime our
sequential setting approximates. Facility location (of which our pool
objective is the thresholded special case) is the standard submodular model
of "representativeness" in data summarization (Lin & Bilmes 2011).

**Packing vs covering, k-center vs k-median.** Farthest-point traversal
(Gonzalez 1985) 2-approximates k-center — the packing objective Part I
paper streams, while k-median/facility-location minimizes *average*
service distance and behaves like our coverage objective (measure-weighted,
bulk-first, outlier-indifferent). Coreset constructions (Har-Peled &
Mazumdar 2004) make the same distinction: an ε-coreset for k-median weights
by mass, a k-center net does not. Our contribution is not new theory for
these objects but the identification of *which one the corpus-construction
product requirement actually is*, plus measurement of what optimizing the
wrong one costs.

**Quality-diversity search.** MAP-Elites (Mouret & Clune 2015) maintains an
"illuminated map": one elite per cell of a hand-designed behavior space —
equal allocation over a fixed grid, in our vocabulary, with quality as the
within-cell objective. Our method differs in that cells are elicited from
the generator and refined recursively rather than fixed, allocation is
marginal-gain-driven rather than one-per-cell, and coverage is measured
against a reachable distribution rather than a designer-set grid; §II.7.5's
equal-allocation row quantifies what the fixed-grid choice costs at small ε.

**Diversity metrics and dedup.** The Vendi score (Friedman & Dieng 2022) is
our spectrum-level diversity readout, computed at O(nD² + D³) via the Gram
trick as in Part I. SemDeDup (Abbas et al. 2023) removes semantic
near-duplicates from web-scale corpora — the deletion-side mirror of our
selection problem: both spend a budget to maximize non-redundant semantic
territory. DPPs (Kulesza & Taskar 2012) give probabilistic diverse selection;
exact DPP sampling scales poorly in n and optimizes a determinant (volume)
rather than covered measure, though greedy MAP-DPP behaves similarly to
facility-location greedy in practice.

**Eval-suite construction.** Perez et al. (2022) generate evaluations with
LMs at scale; red-teaming work (Ganguli et al. 2022) documents exactly the
clustering-into-attack-families failure that motivates a coverage objective
for safety suites.

**Synthetic instruction corpora, and the baselines we compare against.**
Self-Instruct (Wang et al. 2023) bootstraps instructions from a seed pool with
few-shot exemplars and a ROUGE-L similarity filter; the released Stanford
Alpaca corpus (Taori et al. 2023, 52,002 items) is its canonical output.
Evol-Instruct / WizardLM (Xu et al. 2023) instead *mutates* existing
instructions with depth and breadth evolution operators, released as
WizardLM_evol_instruct_V2 (143,000 items). Persona-Hub (Ge et al. 2024)
conditions generation on a catalogue of ~1B personas, releasing 50,000
persona-synthesized instructions; it is the closest published relative of
latent-axis conditioning, differing in that its catalogue is mined at scale
and flat, where our axes are elicited from the generator and refined into a
tree. AttrPrompt (Yu et al. 2023) similarly conditions on attribute
dimensions. We compare against the released artifacts of the first three
(§II.5), rather than against reimplementations, since a reimplementation can be
weak in ways that flatter us. On the selection side our baselines are the
standard ones for diverse subset choice: SemDeDup (Abbas et al. 2023),
farthest-point traversal (Gonzalez 1985), and greedy MAP inference for DPPs
(Kulesza & Taskar 2012; Chen, Zhang & Zhou 2018).

### II.9 Relation to Part I

The two papers are a packing/covering dual pair, and the duality is the
k-center/k-median one:

| | infinite horizon (Part I) | finite budget (Part II) |
|---|---|---|
| horizon | unbounded stream | known n = 10,000 |
| objective | max-min gap + typicality anchor | ε-ball covered measure of μ |
| classical problem | k-center / packing | max coverage / facility location |
| optimum's habitat | boundary, extremes | bulk, measure-weighted |
| where the optimum sits | boundary and extremes | wherever the measure is |
| planning | impossible (no n) | allocation across cells (§II.3.6) |
| guarantee | none exact (packing is 2-approx streaming) | (1 − 1/e), ½ streaming |
| refinement's role | capacity, so the stream never exhausts | reach, at the cost of a moving denominator |

The method stack transfers almost unchanged. Elicited axes, judge gate,
ledger and recursive refinement are objective-agnostic scaffolding for
steering a generator, and only the selection rule and its state differ (running
second-moment matrix and min-gap scales there; reference pool and covered
mask here). Part I's measurements carry over to this side of the duality.
Its psychometric corpus is 72.3% exact duplicates under naive prompting and
0.0% under axis conditioning at the same budget; its poetry pilot moves median
nearest-neighbour distance from 0.089 to 0.239 while cutting 4-gram
self-repetition by a factor of 37. Whatever the objective, those numbers are
about the proposal distribution rather than about the selection rule, and a
coverage pipeline drawing from the unconditioned distribution inherits the same
ceiling.

Two of its findings bear on coverage directly. Refinement pays only when the
children move transverse to the occupied span; children displaced isotropically
near a saturated parent concentrate mass where the corpus already sits, and
§II.7.8 reproduces that under the coverage objective, where the same move buys
density instead of reach. And a metric that rewards unspent headroom rewards a
policy that covered nothing, since a corpus concentrated in one small region
leaves the whole rest of the space available. Every table in §II.5 therefore
reports covered fraction jointly with a within-corpus diversity measure, and
coverage is measured against the reachable manifold's measure rather than
against whatever happens to be left over.

The reverse direction has a caution of its own. Part I's steered image corpus
remains fonder of tiled grids than a human art director would be, because a
packing objective says only that items should sit far apart and says nothing
about where in the space they ought to sit. Coverage carries a reference
measure and so inherits a pull toward where real examples are. That is the term
packing lacks, and the reason the two objectives need each other more than
either needs a better optimizer.

The remaining difference is epistemic. The infinite-horizon side
never has to say what fraction of anything it achieved, while a coverage
claim is *always* a ratio, which is why the denominator has to be named
only on this side of the duality. Conversely, the budgeted side gets
something the unbounded side cannot have: a completion semantics. A coverage
run can report "we are 62% done at resolution ε and the marginal item now
buys 0.1% " — a spend/stop signal no packing objective provides.

### II.10 Limitations

- **ε is chosen, not learned.** All guarantees are per-ε; our calibration
  (quantiles of reachable NN distance) is a heuristic, and §II.7.5 shows the
  cost of optimizing at the wrong radius. A multi-resolution objective
  (integrating coverage over an ε prior) is the obvious next step.
- **The pool is the measure.** Everything is relative to the reachable pool;
  behaviors the base generator cannot emit are invisible until refinement
  opens them (§II.4), and pool refresh is only as good as the refinement
  trigger.
- **Selection bias on the estimation pool.** The head-to-head is scored on a
  reference half no pipeline reads, but the live pilot's selection and
  reporting share one 240-item pool, with the bias direction stated; a larger
  live run should split them.
- **Single embedder, single similarity geometry.** Every coverage number here
  is computed in `nomic-embed-text` space with cosine distance. The parent
  project measured a correlation of only r = 0.155 between text-embedding and
  CLIP-image-embedding pairwise similarity on rendered DALL-E outputs, which
  means text-space coverage is close to blind to whether the *rendered
  artifacts* are diverse. For any domain with a downstream rendering step, the
  objective should be defined in the space the artifact actually lives in; we
  did not do that here.
- **Matched-n does not neutralize generator quality.** In the instruction
  head-to-head (§II.5) the released corpora are 20-70x larger than ours and
  were produced by different, mostly stronger generators, with human curation
  in at least one case. Matched-n sampling controls for size; it cannot
  control for the model that wrote the items, and PersonaHub's persona
  catalogue is orders of magnitude larger than any axis lattice we elicit.

### II.11 Conclusion

At a finite budget, corpus construction is a covering problem, and treating
it as one pays: the coverage functional is exactly submodular in its
Monte-Carlo form, greedy selection is near-optimal in theory and measurement,
and the resulting suites blanket the reachable space instead of decorating
its boundary. Packing (the right shape for an unbounded stream) is
measurably the wrong one here, sacrificing 2–4× covered fraction to win a
min-gap metric no budgeted product requirement asks for. Part I's stack
scaffolding (elicited axes, gates, ledger, recursive refinement) transfers
whole; what does not transfer is innocence about denominators: a method that
grows the space it is covering must say so when it reports how much of that
space it covered.

---

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
