# Review Intelligence

[Français](README.fr.md) · **English**

Automatically sort negative customer reviews by complaint reason, to route each one to the right team.
Three approaches compared on quality, cost and latency: TF-IDF + logistic regression, fine-tuned DistilBERT, and a
Mistral LLM called through the API. Then a costed recommendation. Part 2: a RAG pipeline that drafts a reply to each
review, grounded in a customer policy, evaluated with a calibrated LLM judge ([below](#part-2-drafting-replies-with-rag)).

Data: Amazon Reviews 2023 (McAuley Lab), Grocery_and_Gourmet_Food category, English-language reviews rated 1 to 3
stars (Y. Hou, J. Li, Z. He, A. Yan, X. Chen and J. McAuley (2024), "Bridging Language and Items for Retrieval and
Recommendation", [arXiv:2403.03952](https://arxiv.org/abs/2403.03952)). The repository contains no review text: only
the IDs and labels of the 300 annotated reviews.

Code comments, CLI messages and the decision log (`docs/CONVENTIONS.md`) are in French.

![Summary](livrable/synthese.png)

## In short

- **Recommendation: `mistral-small-2603` with prompt v4**, for batch (non-real-time) sorting.
- **Quality** on 300 hand-annotated reviews: macro-F1 0.770 [0.716, 0.816], exact match 0.763 [0.717, 0.810].
  For reference, the annotator agrees with themself only 0.70 of the time [0.58, 0.82] (second pass against first
  pass, on 50 of these reviews): the task is ambiguous. That reference is measured against a different target and on
  fewer reviews, so it is not directly comparable with 0.763.
- **Actual cost: ~€4.6 per 100,000 reviews** (provider's invoice; prompt caching cuts the list price by 2.7×).
  Latency: 0.76 s per review.
- **DistilBERT**, trained on 4,979 reviews labelled by the LLM: macro-F1 0.712, gap with the LLM −0.058
  [−0.115, +0.003], not demonstrated on the primary metric. Two gaps are demonstrated (paired intervals excluding 0):
  exact match (−0.080 [−0.133, −0.030]) and recall on food-safety complaints (0.70 vs 0.89; −0.196 [−0.340, −0.069]),
  i.e. ~3,000 more safety complaints missed per 100,000 reviews (between ~1,000 and ~5,000).
- **Conditions**: pinned model version; 100 reviews annotated each month to track quality; DistilBERT kept as a
  fallback (API outage, a client whose data must not leave its premises).
- **What would change the recommendation**: a confidentiality constraint, a real-time requirement, or a volume on the
  order of a million reviews per month (threshold to be costed against hosting an in-house model).

## The problem

A review can carry several reasons (multi-label). List v2, validated on 200 exploration reviews:

| Label | Definition | Team (example) |
| --- | --- | --- |
| `reception` | shipping or package: delay, loss, broken or melted product, damaged packaging | logistics |
| `gout_texture` (taste/texture) | taste, texture or smell of a product that arrived intact and not expired | product quality |
| `fraicheur_securite` (freshness/safety) | expired, stale, mouldy, insects, foreign body, health effect | food safety |
| `non_conforme` (not as described) | does not match the description: wrong product, size, undisclosed ingredient | catalogue |
| `prix` (price) | too expensive, poor quantity/price ratio, price increase | pricing |
| `autre` (other) | no complaint matching the reasons above; never combined | customer service |

Full definitions and boundaries (in French): [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) and the annotation guide
[`docs/guide_annotation.md`](docs/guide_annotation.md).

## Results

Test set: 300 randomly drawn reviews, hand-annotated before any prediction, never seen by a prompt or a training run.
95% confidence intervals by bootstrap (2,000 resamples of reviews).

| Model | macro-F1 | micro-F1 | Exact match | Safety recall | Cost per 100,000 reviews | Latency per review |
| --- | --- | --- | --- | --- | --- | --- |
| mistral-small + prompt v4 | **0.770** [0.716, 0.816] | 0.843 | **0.763** | **0.89** | ~€4.6 (invoice; $14.6 without cache) | 0.76 s (API) |
| DistilBERT | 0.712 [0.657, 0.758] | 0.788 | 0.683 | 0.70 | ~€0 + hosting | 7.6 ms (T4 GPU) |
| TF-IDF + logistic regression | 0.634 [0.578, 0.683] | 0.739 | 0.607 | 0.63 | ~€0 + hosting | 0.95 ms (CPU) |
| Reference: annotator, 2nd pass vs 1st (50 of the 300 reviews) | – | – | 0.70 [0.58, 0.82] | – | – | 18 s |

The annotator reference is not directly comparable with the models: it compares the annotator's second pass with
their first, on 50 reviews, whereas the models are measured against the adjudicated reference on 300 reviews; its
interval also contains 0.763. It only shows that the task is ambiguous.

Paired gaps (paired bootstrap on the same reviews):

| Comparison | macro-F1 | Exact match | Safety recall |
| --- | --- | --- | --- |
| DistilBERT − LLM (primary comparison) | −0.058 [−0.115, +0.003] | −0.080 [−0.133, −0.030] | −0.196 [−0.340, −0.069] |
| TF-IDF − LLM | −0.137 [−0.200, −0.072] | −0.157 [−0.213, −0.100] | −0.261 [−0.395, −0.135] |

Safety recall was singled out after the TF-IDF test run, which showed its largest gap on this label: it is a post-hoc
choice among 6 labels. With a Bonferroni correction (1 − 0.05/6 intervals), the gaps stay negative: [−0.383, −0.022]
for DistilBERT, [−0.452, −0.103] for TF-IDF. At the threshold used (0.5), the LLM is also more precise on this label
(0.91 vs 0.78); lowering DistilBERT's threshold would raise its recall at the cost of precision (not tried).

F1 per label (support in parentheses):

| Label | LLM | DistilBERT | TF-IDF |
| --- | --- | --- | --- |
| reception (42) | 0.886 | 0.810 | 0.786 |
| gout_texture (171) | 0.909 | 0.877 | 0.874 |
| fraicheur_securite (46) | 0.901 | 0.736 | 0.652 |
| non_conforme (24) | 0.560 | 0.528 | 0.440 |
| prix (39) | 0.800 | 0.771 | 0.675 |
| autre (32) | 0.566 | 0.552 | 0.375 |

**Reading.** The LLM leads on every label. DistilBERT closes much of TF-IDF's gap, especially on `autre` and `prix`,
but not on safety: it still misses 14 of the 46 freshness complaints, against 5 for the LLM; the LLM catches 10 that
DistilBERT misses, while the reverse happens once. The two labels that are hard for every model (`non_conforme`,
`autre`) are those whose boundary had to be settled by a rule in the guide ("disclosed by default", "vague
complaint").

## Method

1. **Label list** checked on 200 exploration reviews with the LLM: a first list was rejected (one label at 77%,
   another at 0%), list v2 kept.
2. **Human reference**: annotation guide frozen before the first test review; 300 reviews annotated blind (without
   seeing predictions); 50 re-annotated to measure consistency (exact match 0.70); disagreements adjudicated. The
   annotations are published: [`data/annotations_test.jsonl`](data/annotations_test.jsonl).
3. **LLM**: prompt v4 (condensed guide and 6 examples from the exploration set), structured output (justification,
   free-text reason, then labels), temperature 0, pinned model. 5,000 training reviews labelled for $0.73 at list
   price. Acceptance criteria set before the trials; deviations documented.
4. **Models trained on the LLM's labels** (distillation): 1,000 validation reviews drawn from the training set;
   settings chosen on validation (20 configurations for TF-IDF, 9 DistilBERT runs over 3 seeds); retraining on the
   4,979 valid reviews; a single test measurement per model.

**Principles held throughout**: frozen test set; primary metric (macro-F1) and primary comparison fixed before the
results; every decision dated and recorded before the measurement in [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md);
confidence intervals everywhere; a gap is only claimed if its paired interval excludes 0.

## What the invoice taught

At list price, labelling cost $14.6 per 100,000 reviews. The actual invoice is ~2.7 times lower: the system prompt,
identical from one call to the next, was served from the provider's cache for 86.7% of input tokens, billed at 10% of
the price. Output (~46 tokens per review) now accounts for half the cost. Levers not tried: batch processing
(advertised at −50%), shorter output.

## Limitations

- Reference built by **a single annotator**; their self-consistency is measured (0.70 on 50 reviews), not whether their
  definitions hold for someone else.
- **300 test reviews**: intervals are ± 0.05 wide; a 0.06 gap between the LLM and DistilBERT remains unresolved.
- Safety recall singled out post hoc (Bonferroni correction applied above); DistilBERT's threshold not tuned for this
  label.
- Prompt v4 kept despite missing 2 of 3 acceptance criteria (documented decision).
- The in-house models learn the **conventions and errors of the LLM** that labelled them.
- DistilBERT latency measured on GPU; hosting cost of in-house models not estimated.

## Part 2: drafting replies with RAG

For a negative review, draft a reply for the customer service team, grounded in the store's customer policy, with the
action to take (refund, replacement, goodwill gesture, safety escalation, escalation, none). The draft is meant to be
checked by a person before it is sent.

**Corpus**: customer policy of a fictional online grocery store, 46 passages of 105 to 157 words with figures (30
days to complain, refund within 5 business days, safety team within 24 hours...), first draft generated by an LLM,
then checked by hand: [`corpus/politique_client.jsonl`](corpus/politique_client.jsonl). The policy is fictional
because Amazon Reviews 2023 contains no seller policy and no seller replies.

**Pipeline**, written by hand without a framework ([`src/rag.py`](src/rag.py)):

1. **Retrieval**: the review is embedded with `mistral-embed`; exact cosine search over the 46 passages (numpy,
   normalised vectors); the 8 closest passages go into the prompt.
2. **Generation**: `mistral-small-2603`, prompt `rag_v5`, temperature 0, structured output. The model first writes
   the most negative sentence of the review, quoted, and the case (safety, no complaint, complaint), then the reply,
   its citations (`[P12]`) and the action. Choosing the case before writing the reply fixed the safety error seen
   with earlier prompts (pit fragments treated as a taste issue).
3. **Checks in code**, without any extra call: cited passages exist, were provided and appear in the text; case
   consistent with the action; condition assumed to be met ("since this was your first purchase"); safety words in
   the review without a safety escalation.
4. **Corrective retry** (version `rag_v5d`): when the case contradicts the quoted sentence, or a safety word or an
   assumed condition is detected, a single retry with the first answer and a message naming the problem. Whatever
   remains is flagged for the reviewer, not fixed.

**Evaluation sets**: drawn once from the reserve, with no review in common with the 300-review test set: `rag_dev`
(50 reviews, for tuning) and `rag_eval` (150 reviews, frozen; the first 50 are used).

| Step | What is measured | Result |
| --- | --- | --- |
| Retrieval | recall@8 (at least one relevant passage in the top 8), 26 `rag_dev` reviews with relevant passages annotated by hand | 0.885 (recall@5: 0.769 [0.615, 0.923]) |
| Human ratings | 20 replies from prompt v4, 3 yes / no criteria: correct action, grounded, tone | 2 acceptable out of 20 |
| LLM judge | `mistral-large-2512` against the 20 human ratings, criterion by criterion | validated on correct action only (κ = 0.79) |
| Generator | correct action according to the judge, v4 → v5d, 50 `rag_dev` reviews, paired | 0.40 → **0.74**, +0.34 [+0.18, +0.50] |
| RAG vs whole policy | correct action, 50 `rag_eval` reviews, paired | 0.68 vs 0.78, gap not demonstrated |

### What the evaluation taught

**Retrieval without a filter.** Restricting the search to passages of the complaint reasons predicted by the part 1
classifier excluded the right passage whenever the classifier was wrong, including a safety passage. Kept: k = 8,
no filter (8 passages instead of 5 cost about $0.00007 more per reply).

**A judge is only trusted where it was checked.** The 20 human ratings came first. The acceptance rule was fixed
before running any judge: for each criterion, recall of faults ≥ 0.80, precision ≥ 0.67, κ ≥ 0.40 (on tone, 14
replies out of 20 were faulty, so a judge saying "faulty" to everything kept a precision of 0.70; κ rules it out).
Three judges were tried (Large 3 with two prompt versions, then the Large 4 preview): none passes all three
criteria. On tone, the judges rate politeness rather than whether the reply addresses the actual problem. Kept:
Large 3 with judge prompt v2, **for the correct action only**; grounding and tone are covered only by the 20 human
ratings and the checks in code.

**A rule the model does not apply goes into code.** With prompt v4, 8 of the 10 replies labelled "no complaint"
answered actual complaints (taste, smell, expiry date, size). Prompt v5, with a stricter definition, did not help
(0.38 vs 0.40): in 22 of 26 such cases, the model quoted a negative sentence and still chose "no complaint". A
retry triggered by code fixed it: 0.70 (v5c, retry on this case only), then 0.74 (v5d, also on safety words and
assumed conditions); 20 reviews won against 3. The adoption rule was corrected after the measurement, openly
(safety escalations counted in both directions), and the correction is recorded in
[`docs/CONVENTIONS.md`](docs/CONVENTIONS.md). Known remaining error: a review mentioning pit fragments, escalated by
v4, gets a store credit from v5d (the reply is flagged for the reviewer).

**RAG or the whole policy in the prompt?** Same generator (`rag_v5d`), same 50 `rag_eval` reviews:

| | RAG (8 passages) | Whole policy (46 passages) |
| --- | --- | --- |
| Correct action (judge) | 0.68 | 0.78 |
| Paired gap, whole policy − RAG | | +0.10 [−0.02, +0.22]; 8 reviews won against 3, sign test p = 0.23 |
| Replies flagged by the code checks | 22% | 8% |
| Input / output tokens per reply | 3,553 / 182 | 10,795 / 183 |
| Median latency | 1.55 s | 1.89 s |

At 46 passages, putting the whole policy in the prompt does at least as well, more simply (no index, no missed
passage, no citation outside the provided passages), and with prompt caching the cost per reply is similar (~$0.0005,
estimated, not checked on the invoice). The service still uses RAG, with the whole-policy mode as an option: the gap
is not demonstrated, RAG sends 3 times fewer tokens (~2.7 times cheaper without cache), answers faster, and its cost
does not grow with the corpus. RAG becomes necessary once the corpus no longer fits the context (~1,500 passages of
this size) or when access rights differ between documents.

### Limitations of part 2

- Fictional policy, short and consistent; a real policy is longer and contradicts itself.
- **A single rater, 20 replies**; the judge prompt was tuned on these same 20 replies (winner's bias), and its
  validity on the action is assumed to carry over from v4 replies to v5d replies.
- **50 reviews per comparison**: intervals are about ± 0.15 wide.
- Retrieval measured on 26 `rag_dev` reviews only, not re-measured on `rag_eval`.
- Grounding and tone of the final version are not measured by a validated judge; 22% of RAG replies are flagged by
  the code checks. Replies are drafts for a person to check.

Next: an API (FastAPI), a Docker image and a Streamlit demo.

## Reproduce

Setup (Windows):

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env          (then set MISTRAL_API_KEY in it)
python -m pytest -q
```

Steps:

```
python -m src.data --n-read 500000                          # pool and split (seed 42) -> data/pool.parquet
python -m src.label_llm --part explo --prompt v4            # LLM labelling (also --part train, --part test)
python -m src.report data/labels_explo_v4.jsonl             # label shares, invalid outputs, cost, latency
python -m src.annotate                                      # manual annotation of the 300 test reviews
python -m src.metrics accord                                # annotator consistency (pass 1 vs pass 2)
python -m src.metrics eval data/labels_test_v4.jsonl        # one model against the reference
python -m src.tfidf_lr grid                                 # TF-IDF: grid on validation
python -m src.tfidf_lr final                                # TF-IDF: single test measurement
python -m src.distilbert paquet                             # DistilBERT: bundle for Colab
                                                            # then notebooks/distilbert_colab.ipynb (T4 GPU)
python -m src.distilbert compare --out-dir FOLDER           # TF-IDF vs DistilBERT on validation
python -m src.metrics eval data/labels_test_v4.jsonl --contre data/preds_test_distilbert.jsonl
python livrable/slide.py                                    # summary slides (English and French, PNG and PDF)
```

Part 2 (RAG):

```
python -m src.rag index                                     # embed the policy -> data/rag/index.npy
python -m src.rag jeux                                      # draw rag_dev (50) and rag_eval (150) from the reserve
python -m src.rag annoter --part rag_dev                    # relevant passages, annotated by hand
python -m src.rag eval-recherche --part rag_dev             # recall@k and MRR
python -m src.rag repondre --part rag_dev --prompt rag_v5 --correction-etendue   # replies (rag_v5d)
python -m src.juge page --part rag_dev                      # HTML page to rate replies by hand
python -m src.juge juger --part rag_dev --prompt rag_v4     # LLM judge on the replies
python -m src.juge calibrer --part rag_dev                  # judge against the human ratings, criterion by criterion
python -m src.juge bilan --part rag_dev --prompt rag_v5d --contre rag_v4   # paired comparison of two versions
python -m src.rag repondre --part rag_eval --mode plein --prompt rag_v5 --correction-etendue --limit 50
python -m src.juge comparer --prompt rag_v5d                # RAG vs whole policy on rag_eval
```

## Layout

| Path | Contents |
| --- | --- |
| `src/data.py` | streaming read, cleaning, pool of 20,000 reviews, explo / test / train split |
| `src/label_llm.py` | labelling through the Mistral API: structured output, rules, retries, resume after interruption |
| `src/report.py`, `src/compare_prompts.py` | label analysis, prompt comparison |
| `src/annotate.py` | command-line annotation tool (passes, doubts, adjudication) |
| `src/metrics.py` | kappa, per-label F1, micro / macro, bootstrap, paired bootstrap |
| `src/tfidf_lr.py` | TF-IDF + logistic regression: grid, final measurement, prediction explanations |
| `src/distilbert.py` | DistilBERT: resumable grid, final measurement, comparison with TF-IDF |
| `notebooks/distilbert_colab.ipynb` | running DistilBERT on Colab |
| `src/rag.py` | RAG: index, retrieval, reply generation, checks in code, corrective retry, recall@k |
| `src/juge.py`, `src/gabarit_notation.html` | LLM judge: rating page, judging, calibration, paired comparisons |
| `corpus/politique_client.jsonl` | fictional customer policy (46 passages) |
| `prompts/` | versioned prompts: classification v1 to v4, replies `rag_v1` to `rag_v5`, judge `juge_v1` and `juge_v2` |
| `tests/` | 114 tests (`python -m pytest -q`) |
| `docs/CONVENTIONS.md` | every decision and every figure, dated (French) |
| `docs/guide_annotation.md` | annotation guide (v1, frozen; French) |
| `livrable/` | summary slides: `slide.py` produces `synthese.png` / `.pdf` (English) and `synthese_fr.png` / `.pdf` (French) |

Code under the MIT licence (see [`LICENSE`](LICENSE)).
