# Research Council — PRD (Layers 1–3)

**Scope of this PRD:** foundations, infrastructure, and orchestration through Round 2 controller. Layers 4–7 (human-in-the-loop, output/UX, persistence, quality/operations, explicit out-of-scope) are deferred to a follow-up PRD.

---

## Problem Statement

I am an AI engineer doing research on open ML problems. When I have a research problem — whether a polished proposal I'm 80% committed to, or a half-formed idea I'm not sure is worth investing in — I currently use a mix of careful Claude prompts, Google Scholar searches, and ad-hoc conversations to pressure-test my thinking. This is slow and uneven. The substitutes do not reliably:

1. Catch unsupported claims in my own framing — I cite numbers and papers from memory, and they're not always right.
2. Surface prior art that I'd otherwise spend weeks rediscovering after committing to an approach.
3. Apply structurally incompatible methodological frames to the same problem in a way that surfaces gaps a single coherent perspective would miss.
4. Resist sycophancy when I push back. A Claude conversation accommodates me; a careful colleague pushes back but isn't always available.

The cost of being wrong on a research direction is at least weeks of work, often months. The cost of the current workflow is missing the kinds of gaps that only show up when something is examined from multiple incompatible angles before I commit. I need a tool that compresses the "would I have found this gap before committing two months to the approach" loop from weeks to one work session.

## Solution

A multi-agent deliberation system, single-user, that takes a research problem (or proposed solution to one) through a structured deliberation:

1. **Intake (Round 0):** I declare whether I'm bringing a polished proposal (`deep_dive` mode) or an early idea (`exploratory` mode). A master agent walks me through producing a structured brief, with parallel verification of any cited claims I include as background.
2. **Independent analysis (Round 1):** A panel of methodological lenses (8 by default in `deep_dive`, 4 by default in `exploratory`) analyzes the brief in isolation, each producing typed findings under a controlled epistemic vocabulary.
3. **Cross-pollination (Round 2):** The adversarial lens, prior-art lens, and any lens that flagged self-disagreement-conditional-on-peers in Round 1 see anonymized Round 1 outputs and produce critique findings.
4. **Synthesis (Round 3) — *deferred to follow-up PRD.***

Two structural commitments make this different from a careful Claude prompt:

- **Per-claim verification against real literature.** Every empirical and prior-art claim — both in my brief and in lens outputs — is checked against Semantic Scholar, arXiv, OpenAlex, and Crossref. Unverifiable claims are tagged, never silently passed through. Hallucinations are filtered at the boundary, not prevented at generation.
- **Structured outputs over typed schemas.** Lenses do not emit prose essays. They emit typed Findings with claim_type, confidence indicator, sources, and failure_modes_if_wrong. This makes hallucinations detectable, makes verification possible per-claim, and makes synthesis operate on typed inputs rather than prose-on-prose.

The killer feature is **gap-finding on a proposed solution**, with verified citations and prior-art grounding — not open-ended brainstorming.

## User Stories

### Scope and success criteria (Layer 1.1)

1. As a researcher, I want the v0 system to demonstrate value on a real currently-open research problem of mine in the first session, so that I don't invest in a tool whose utility is purely hypothetical.
2. As a researcher, I want explicit abandon triggers tied to observable failure conditions, so that sunk-cost reasoning cannot defend a non-functional v0.
3. As a researcher, I want an abandon trigger of >10% verification fail rate (citations that don't exist, don't say what was claimed, or can't be located), so that the verifier cannot quietly become a credibility-laundering machine.
4. As a researcher, I want an abandon trigger when the council, on my Active Context Inhibition proposal specifically, surfaces fewer than 2 of the 5 gaps I already privately identified, so that architectural failure is detectable on the first real run.
5. As a researcher, I want an abandon trigger when total user-side attention for one deliberation cycle exceeds 60 minutes, so that the council is forced to be faster than my current substitutes (careful Claude prompt + 30 min of literature search).
6. As a researcher, I want system-side latency and cost to be unconstrained, so that the council can do expensive background work (multi-stage verification, multiple LLM calls per lens, full Round 1+2 cycle) without optimization pressure compromising quality.
7. As a researcher building v0 alone, I want "v0" to mean "the version I'd actually use and trust on real research," not "minimal demo" — features ship if they are real-use-essential, deferred if they are large enough to warrant their own version.

### Data model (Layer 1.2)

8. As a system, I need a Session object as the top-level container for one research problem, with status (intake / consulting / synthesized / iterating / closed), so that all rounds and outputs are scoped to a single research thread.
9. As a system, I need a Brief object that is versioned (immutable post-confirmation, edits produce new versions), so that LensRuns are reproducibly scoped to a specific brief state.
10. As a system, I need a LensRun object as the atomic unit of work, scoped to (lens identity, round, brief version, dispatch event), so that re-runs and partial re-runs can be tracked.
11. As a system, I need a Finding object as the unit of synthesis (not LensRun), with claim_text, claim_type, confidence, sources, failure_modes_if_wrong, verification_status, so that synthesis operates on typed claims rather than prose.
12. As a system, I need a VerificationResult object attached to a Finding (or to a citation within a Finding), with what was checked, against what source, the verdict, and the verifier's evidence, so that re-verification is possible and history is auditable.
13. As a system, I need a Source object as a first-class deduplicated entity (identity = DOI / arXiv ID / OpenAlex ID / hash of normalized title+first author+year), so that the verifier can cache and reuse source data across Findings, LensRuns, and Sessions.
14. As a system, I need a Challenge object capturing user↔lens re-engagement (challenged LensRun, user challenge text, lens response), so that iteration loops have first-class data representation.
15. As a system, I need a Synthesis object referencing (not copying) Findings, with agreement map / tension map / gap list / conditional recommendations, so that downstream changes to Findings propagate.
16. As a system, I need a Verdict object modeled now (researcher's later-stated truth about what worked, supporting evidence, predictions_validated, predictions_invalidated as Finding pointers) even though no UI/loop in v0, so that historical session data is recoverable as eval material later.
17. As a system, I need a DispatchEvent object capturing (brief_version, panel, timestamp, lens_run_ids), so that re-runs of the same brief with different panels stay distinguishable without bumping brief version.
18. As a system, I do NOT need a persisted Agent object — lenses are reconstructed each round from (brief, this lens's prior LensRuns on this brief, challenges to this lens) — so that anti-sycophancy is structurally enforced via statelessness.
19. As a system, I do NOT need a Round object — round number is a field on LensRun, round-level orchestration lives in code — accepting that round-level metadata (e.g., "Round 2 skipped due to insufficient Round 1 success") gets stored on Session if needed.
20. As a builder, I want to commit to writing one manual Verdict against the Active Context Inhibition session within 90 days of running it, so that the Verdict schema is exercised at least once and doesn't drift into unused-schema rot.

### Epistemic schema (Layer 1.3)

21. As a Finding, I have one of five claim_types — `empirical`, `prior_art`, `mechanism_hypothesis`, `gap`, `failure_mode` — each with distinct verification semantics.
22. As an `empirical` claim, I assert "X paper reports Y result on Z benchmark" and am verifiable against literature with the highest verifier confidence.
23. As a `prior_art` claim, I assert "work X exists that does Y" without a specific numerical claim, and am verifiable with a lower bar than `empirical` (existence + characterization, not specific numbers).
24. As a `mechanism_hypothesis` claim, I assert "I think the cause is Z" and am unverifiable against literature — verifier checks only that I'm tagged correctly, not that I'm right.
25. As a `gap` claim, I assert "the proposal doesn't address W" — verifier checks against the brief, not external sources.
26. As a `failure_mode` claim, I assert "if assumption A is wrong, then consequence B" — verifier checks logical structure, not external sources.
27. As a Finding, I have one of three confidence indicators — `load_bearing` (lens stakes its overall analysis on this), `supporting` (lens believes and uses this, but analysis survives if invalidated), `exploratory` (lens floats this for consideration, would not defend) — and these are per-lens per-claim, NOT system-level.
28. As a system, I compute cross-lens agreement structurally at synthesis time ("7 of 10 lenses independently flagged this; 3 marked it load_bearing, 4 supporting"), never as a numerical confidence percentage.
29. As a VerificationResult, I have one of seven statuses: `unverified` (verifier hasn't run yet — explicit state, not absence), `verified`, `partially_supported` (source relevant but claim overstated/imprecise), `contradicted` (source actively contradicts), `source_not_found` (citation doesn't resolve), `unverifiable_by_design` (claim type isn't externally checkable — not a failure state), `verifier_error` (verifier couldn't run, with sub-reason).
30. As a VerificationResult, I keep `partially_supported` and `contradicted` as separate states because they imply different downstream actions (sloppy vs. wrong).

### Claim verifier (Layer 2.1)

31. As a verifier, I query Semantic Scholar and arXiv as co-primary sources in parallel, OpenAlex as secondary, Crossref as fallback for DOI resolution.
32. As a verifier, when both S2 and arXiv return a hit for the same source, I apply a tiebreak: prefer source with full text available; if both have full text, prefer S2 (better-structured metadata); if neither has full text, prefer arXiv.
33. As a verifier, I run a 3-stage pipeline per claim: Stage 1 source resolution, Stage 2 locality check (find passages in source bearing on claim), Stage 3 entailment check (passage supports, contradicts, or partially supports claim).
34. As a verifier, I keep Stage 2 and Stage 3 separate (not fused) because their failure modes are distinct: Stage 2 failure means "paper doesn't address this," Stage 3 failure means "paper addresses this but the claim summarized it wrong" — synthesis needs the distinction.
35. As a verifier, for `mechanism_hypothesis`, `gap`, and `failure_mode` claims (tagged `unverifiable_by_design`), I run a tag-appropriateness check via single LLM call with no source lookup, confirming the claim text matches its declared tag — so lenses cannot dodge verification by mistagging empirical claims as hypotheses.
36. As a verifier, I cache at three levels: Source cache (keyed on canonical_id, indefinite with 24h metadata refresh), Locality cache (keyed on source_canonical_id + claim_text_hash, indefinite), Entailment cache (keyed on source_canonical_id + claim_text_hash + claim_type, indefinite), with a `verifier_version` field on cache keys so prompt/model upgrades invalidate cleanly.
37. As a verifier, I never silently turn `verifier_error` into `verified`. When I can't reach a source, the claim stays unverified and synthesis is informed.
38. As a verifier, I handle failures with explicit sub-reasons: `rate_limited` (retry with backoff, 3 attempts), `paper_paywalled_no_abstract` (permanent failure for that claim against that source), `api_down` (retry-later-bulk, not immediate), `parse_error` (single retry then persist).
39. As a system, I include `verifier_error`-tagged claims in synthesis with a visible "unverified" tag, never strip them silently, never block synthesis pending retry.
40. As a verifier, I make hallucination detectable structurally: Stage 3 output MUST include a verbatim passage quotation from the source; before persisting, a string-match check confirms the quoted passage actually appears in the source text; if it doesn't, the result is downgraded to `verifier_error` with sub-reason `passage_hallucinated`.
41. As a builder, I commit to random-sample manual auditing of 5% of verification results for the first month of dogfooding, so that "verifier might quote a real passage that doesn't actually support the claim" remains under observation.

### Sub-agent contract (Layer 2.2)

42. As a LensRun, I receive exactly four inputs: `brief` (current version, full, including claim-level verification status), `lens_config` (identity, frame, characteristic move, tool access, role, mode-aware behavior), `prior_self` (this lens's prior LensRuns on this brief, as typed Findings only — no prompts, no reasoning, no verifier results), `cross_pollination` (anonymized peer Round 1 outputs, empty in Round 1).
43. As a LensRun, I do NOT receive the master agent's reasoning about why I was selected — only the brief itself — so that the master cannot bias my analysis.
44. As a LensRun, I emit a typed output with: `findings` (list of Finding objects), `open_questions` (list of strings — questions I can't answer from the brief but think matter), `disagreements_with_my_own_framing` (mandatory, minimum 1, list of strings), `brief_summary` (single paragraph max 150 words, how I summarize the brief in my frame), `refused` (optional boolean with reason if I decline analysis).
45. As a LensRun, if I produce zero entries in `disagreements_with_my_own_framing`, that is a schema violation — this is a forcing function against sycophantic over-confidence.
46. As a LensRun, I may legitimately refuse to analyze a brief that's out of my frame, by setting `refused: true` with a reason — refusal is a real outcome handled gracefully by the master, not a failure.
47. As a LensRun, my tool access is declared in `lens_config` and enforced by the runner (not by my prompt). v0 tools available: `verifier_query`, `source_fetch`. Web search, code execution, and arbitrary HTTP are explicitly NOT available in v0.
48. As a system, when a lens calls the verifier, gets `source_not_found`, and then emits the claim anyway as `mechanism_hypothesis`, I log this pattern as an audit signal — so synthesis can detect this evasion pattern.
49. As a LensRun, my prior_self contains only typed Findings — not prompts, not internal reasoning, not the verifier results on those findings. To carry reasoning across rounds, I must have put it in structured outputs in the earlier round.
50. As a system, schema validation on LensRun output is two-strike with no auto-repair: if invalid, send back to the lens with the validation error and ask to fix (one retry); if still invalid, persist as `status: schema_invalid` and exclude from synthesis with explicit note.
51. As a system, I do NOT tolerate partial validity — if some findings are valid and some are invalid, the whole LensRun is invalid, so that sloppy lens prompts can't pass.
52. As a system, the following are enforced by code (not trusted to prompt): output schema validity, tool access scope, round membership (Round 2 lenses get cross_pollination, Round 1 lenses never do), anonymization in cross_pollination (lens identities are stripped at input-construction layer, replaced with `Lens A` / `Lens B`).
53. As a system, the following are enforced by prompt only (with eval as the verifier): lens staying in frame, finding quality and non-triviality, honest filling of `disagreements_with_my_own_framing`, calibrated confidence indicators.

### Lens roster (Layer 2.3)

54. As a system, I ship with 9 lenses in v0: prior-art, adversarial, empirical benchmarking, mechanistic interpretability, information-theoretic, training-data/distribution, deployment/serving, architecture, first-principles. Cognitive science is cut for v0.
55. As the prior-art lens, my frame is "what existing work has already addressed this problem or attempted this solution," my characteristic move is searching literature and surfacing closest prior art the researcher hasn't cited, my role is grounder, and I have `verifier_query` + `source_fetch` tool access.
56. As the adversarial lens, my frame is "this proposal is wrong; my job is to find the strongest reason why." In Round 1 I generate failure_mode findings against the proposed solution; in Round 2 I generate failure_mode findings against the *emerging Round 1 consensus*. My role is critic, with `source_fetch` access. I run in BOTH rounds with different jobs.
57. As the empirical benchmarking lens, my frame is "what would the experiment actually look like and is the proposed evaluation sufficient," I scrutinize missing baselines / ablations / comparisons, my role is critic+grounder, with `verifier_query` + `source_fetch` access.
58. As the mechanistic interpretability lens, my frame is "what's happening inside the model that causes the phenomenon at the level of circuits, attention patterns, MLP activations," my role is generator+critic, no tool access in v0 (my output is mechanism_hypothesis and failure_mode, not prior_art).
59. As the information-theoretic lens, my frame is "what does the problem look like through compression, entropy, mutual information, channel capacity," my role is generator, no tool access. Constraint to prevent mathematical decoration: my findings MUST make at least one quantitative or conditional prediction, not just reframe.
60. As the training-data/distribution lens, my frame is "what does the model's training distribution look like and does the problem/solution interact with that distribution," my role is generator+critic.
61. As the deployment/serving lens, my frame is "what does this look like when deployed: memory, latency, KV cache, serving constraints," I catch "solution doubles inference cost and brief doesn't mention it," my role is critic. (Renamed from "systems/inference-time" for clarity.)
62. As the architecture lens, my frame is "what architectural changes would address the problem, considering the design space of attention variants / position encodings / retrieval-augmented architectures / mixture-of-experts," my role is generator+critic.
63. As the first-principles lens, my characteristic move is taking the proposed solution and asking "what is the minimum mechanism required to produce the observed phenomenon, ignoring all engineering convenience and inherited architectural choices" — generating counterfactual reframings. My output is EXCLUSIVELY `mechanism_hypothesis` and `gap` findings; I am explicitly FORBIDDEN from `empirical` or `prior_art` claims. This restriction is non-negotiable and is the discipline that keeps this lens from drifting into woo.
64. As a system, lens build order is gated, not deployment-prioritized: verifier → tier 1 (prior-art, adversarial) tested on the ECF brief and producing trustworthy output → tier 2 (the other 7 lenses) built and shipped in parallel. At deployment all 9 are peers in the panel.
65. As a system, every brief gets verified before dispatch — the master surfaces verification problems at the end of intake, the user can fix them, replace them, or dispatch with claims explicitly tagged `unverified` (lenses see the tags).

### Master agent — intake (Layer 3.1)

66. As a user starting intake, the master's first message asks me to declare mode: deep_dive (mature proposal to pressure-test) or exploratory (early idea to explore). Explicit, one-tap, not inferred.
67. As a user in deep_dive mode, the master walks me through all 9 brief fields, expects all 9 present before considering termination, and runs brief verification on `background_claims` in parallel.
68. As a user in exploratory mode, the master holds a lower bar: 4 fields minimum (`problem_statement`, one of (`proposed_solution` / `researcher_context`), `success_criteria_for_deliberation`, `scope_and_non_scope`), 2-turn stability, no self-test, often-empty `background_claims` so brief verification is a no-op.
69. As a Brief, I have 9 fields in this elicitation order: `problem_statement`, `proposed_solution` (optional), `researcher_context`, `background_claims` (typed claims with sources, runs through verifier), `success_criteria_for_deliberation`, `scope_and_non_scope`, `panel_constraints` (optional), `prior_panel_consultations` (refinement only), `mode` (enum: exploratory / deep_dive).
70. As a user in deep_dive mode, intake termination requires: all 9 fields present AND no field edited in last 2 intake turns AND master runs a self-test (generates 3 questions a lens might ask that can't be answered from the brief; surfaces them as a recommendation, not a gate).
71. As a user in deep_dive mode, the self-test is a recommendation surface only — I can always type "dispatch" and the master dispatches with gaps explicitly flagged in the brief.
72. As a master agent in early intake (first 3 turns), I ask open questions rather than drafting field contents — I don't have basis for drafts and would anchor the user.
73. As a master agent after the first 3 turns of intake, I propose drafts for every field except one — accept/edit/reject — for speed.
74. As a master agent, I NEVER draft `success_criteria_for_deliberation`. That field must come from the user in their own words because it determines what "useful" means downstream.
75. As a user confirming a brief, I see two screens: Screen 1 is the full brief in schema order (edit any field inline, no verification results); Screen 2 is dispatch readiness summary (self-test questions and bearing fields, verification results on background_claims, proposed lens panel with one-sentence reasoning per included AND excluded lens). I confirm or return to Screen 1.
76. As a system, brief versions are immutable post-confirmation. Any edit produces a new version. The master cannot silently edit a confirmed brief; "dispatch but actually change X first" produces Brief v2 and re-shows Screen 2.
77. As a Lens, my mode (`exploratory` / `deep_dive`) is part of the brief I receive, and I adjust my output accordingly: in exploratory mode, more `exploratory`-confidence findings, fewer `load_bearing`, even more emphasis on `disagreements_with_my_own_framing` because I'm reasoning on thinner inputs.
78. As a master agent, I decline dispatch in 3 categories: (a) not a research question, (b) insufficient specificity even after intake, (c) brief contains claims that all failed verification (>3 unverified, 0 verified). I do NOT decline for "out-of-research-domain predictions" — this is single-user v0 and the user manages that themselves.
79. As a master agent, when brief verification surfaces problems, I present them at the end of intake (not blocking the conversation) with three options per problem: fix, replace, or dispatch-with-`unverified`-tag.
80. As a BriefVerificationStream, I run silently in the background during intake — verification of `background_claims` happens in parallel with the intake conversation, so by the time the user is ready to dispatch, verification has finished.

### Master agent — routing (Layer 3.2)

81. As a master agent, when the user has not pre-specified the panel, I propose a panel via hybrid routing: I analyze the brief and propose a panel with one-sentence reasoning per included AND excluded lens, the user confirms or edits during Screen 2.
82. As a master agent, my routing rationale is per-brief-specific, not boilerplate — exclusion reasoning matters as much as inclusion reasoning, so the user can spot wrong calls.
83. As a master agent in exploratory mode, my default panel is narrower (4-ish lenses), but I still produce inclusion + exclusion reasoning across all 9 so the user can see who I didn't pick and why — and override if they want.
84. As a system, routing logic is identical across modes; only the defaults shift.
85. As a user, when I edit the panel during Screen 2 confirmation, this is metadata on a new DispatchEvent, NOT a brief version bump. The brief content is unchanged; the dispatch configuration is what changes.
86. As a user, when I edit the `panel_constraints` field DURING intake (before confirmation), this IS a brief edit — the distinction is dispatch-time edits vs. brief-content edits.

### Round 1 controller (Layer 3.3)

87. As a Round1Controller, I dispatch all selected lenses in parallel (no staging within Round 1) — lenses are isolated by design so there's no information flow reason to serialize.
88. As a Round1Controller, I rely on the verifier service's internal rate-limit handling (queueing, backoff) — lenses see a verifier that appears to handle high concurrency; the verifier serializes internally.
89. As a Round1Controller, I enforce two timeouts per LensRun: hard ceiling 10 minutes wall-clock (kills the LensRun and persists `status: timeout`), no-progress timeout 3 minutes (catches stuck-mid-call where API isn't returning errors but isn't producing tokens).
90. As a Round1Controller, I do NOT auto-retry timed-out LensRuns — a timed-out run is likely infrastructure or prompt issue, and auto-retry without addressing cause burns time. User can manually re-run via refinement loop (Layer 4.2, deferred).
91. As a Round1Controller, when a LensRun fails schema validation after the 2-strike retry, I surface it explicitly in synthesis output: "8 lenses dispatched; 7 produced valid output; 1 (architecture) failed schema validation after retry — excluded from synthesis. Raw output preserved for debugging."
92. As a LensRun encountering a verifier failure mid-run, I persist the verifier result as-is into the Finding's `verification_status` field and continue. I do NOT retry the verifier myself (the verifier has its own retry logic), I do NOT adapt my output based on verifier failure, I do NOT silently drop the unverified claim.
93. As a Round1Controller, I consider Round 1 complete only when every dispatched LensRun has reached a terminal state (`succeeded`, `schema_invalid`, `timeout`, or `refused`) — no partial proceeds.
94. As a Round1Controller, if more than 70% of LensRuns reach a non-succeeded terminal state, I halt before advancing to Round 2 and surface to user: "N of M lenses failed; this is unusual. Investigate before proceeding?"

### Round 2 controller (Layer 3.4)

95. As a Round2Controller, mandatory participants are the adversarial lens (self-anonymized) and the prior-art lens (sees Round 1 to identify prior art relevant to the council's emerging analysis, not just the original brief).
96. As a Round2Controller, optional participants are any tier 2 lens whose Round 1 output flagged at least one `disagreements_with_my_own_framing` entry that explicitly references "would change if other lenses said X" — narrow opt-in, not full panel re-run.
97. As a Round2Controller, anonymization scheme: lenses become `Lens A`, `Lens B`, … in the order Round 1 completed (not in any meaningful order). Adversarial sees its own Round 1 as one of the anonymized peers with no distinguishing marker, forcing fresh engagement.
98. As a Round 2 LensRun, I receive from Round 1 peers: typed Findings, `brief_summary` (useful for seeing how each lens framed the brief), `open_questions`. I do NOT receive `disagreements_with_my_own_framing` from peers — that field is for synthesis, not for cross-pollination, because exposing it invites social-style "I see Lens A doubts itself, I'll attack there" rather than substantive critique.
99. As a Round 2 Finding, I have the same schema as Round 1 Findings plus two extra fields: `round: 2` and `responding_to` (list of anonymized lens IDs, if the finding is specifically engaging with peer outputs).
100. As a Round2Controller, failure handling is identical to Round 1 (10-min timeout, two-strike validation, schema-invalid surfaced). If more than 70% of Round 2 LensRuns fail, synthesis runs on Round 1 only with a note "Round 2 cross-pollination failed." Round 2 failure does NOT halt synthesis.
101. As a Round2Controller, I skip Round 2 entirely if Round 1 finished with fewer than 3 successful LensRuns (no meaningful consensus to critique), and synthesis runs directly on Round 1 with the skip noted.

## Implementation Decisions

### Domain glossary

The PRD uses these terms with these meanings throughout:

- **Council** — the system as a whole.
- **Lens** — a methodological frame, configured statically (a type, not a runtime instance).
- **LensRun** — one execution of a Lens against a specific Brief version in a specific Round (the runtime instance).
- **Brief** — the structured artifact produced by intake, versioned, immutable post-confirmation.
- **Round 0 / 1 / 2 / 3** — intake / independent analysis / cross-pollination / synthesis. Round 3 deferred to follow-up PRD.
- **Finding** — a typed claim emitted by a LensRun, the unit of synthesis.
- **VerificationResult** — verifier's check on a Finding (or citation within), attached to the Finding.
- **Source** — first-class deduplicated paper/source entity.
- **DispatchEvent** — a (brief_version, panel, timestamp) tuple captured when a Brief is dispatched, distinguishing re-runs with different panels.
- **Verdict** — researcher's later-stated truth about what worked, attached to a Session.
- **Challenge** — user↔lens re-engagement (Layer 4, deferred).
- **deep_dive / exploratory** — intake modes.

### Module sketch (deep modules to build/test in isolation)

The following modules will be built. Each is deep in the sense of "encapsulates a lot of functionality behind a simple interface that rarely changes":

1. **IntakeOrchestrator** — runs the master agent's intake conversation. Owns mode selection, brief schema enforcement, termination criteria per mode, draft-vs-ask logic, two-screen confirmation, the 3 decline gates.
2. **BriefVerificationStream** — runs in parallel with intake on `background_claims`. Owns parallel-verification-during-intake behavior, end-of-intake problem surfacing, carry-unverified-tags-into-dispatch behavior.
3. **ClaimVerifier** — the 3-stage pipeline (resolve / locality / entailment), tag-appropriateness check for unverifiable-by-design claims, 3-level cache with version invalidation, hallucination-detection via passage-quote string match, failure handling with sub-reasons.
4. **PanelRouter** — proposes panel with per-lens inclusion + exclusion reasoning, honors `panel_constraints` overrides, mode-aware defaults.
5. **Round1Controller** — parallel dispatch, timeout enforcement (10min hard / 3min no-progress), schema validation handling with 2-strike retry, completion detection, >70% failure halt.
6. **Round2Controller** — Round 2 participant selection (mandatory + opt-in), anonymized cross-pollination input construction, dispatch, failure handling, skip-condition logic.

Cross-cutting data structures (not modules with logic, but explicit objects in the data model): Session, Brief (versioned), LensRun, Finding, VerificationResult, Source, DispatchEvent, Challenge, Synthesis, Verdict.

### Key architectural commitments

- **State lives in the brief, not in long-running agent conversations.** Agents are reconstructed each round from (brief, this lens's prior LensRuns on this brief, challenges to this lens). No persisted Agent object.
- **Hallucinations are filtered at the boundary, not prevented at generation.** Claim verifier as core infrastructure. Verification status is a real field; `verifier_error` never silently becomes `verified`.
- **Anonymization is enforced in code at input construction, not in prompt.** Lens identities are literally not in the data passed to Round 2 lenses, replaced with `Lens A` / `Lens B`.
- **Schema validity is enforced in code; quality is enforced in prompt with eval as the verifier.** Two-strike validation, no auto-repair, no partial validity.
- **Brief versions are immutable post-confirmation.** Edits always produce new versions.
- **Confidence is structural, not numerical.** Per-lens per-claim ordinal indicators (load_bearing / supporting / exploratory); cross-lens agreement computed at synthesis, displayed as counts not percentages.

### Schemas (decision-encoding snippets)

**Finding** — emitted by lenses, the unit of synthesis:

```
Finding {
  claim_text: string
  claim_type: enum { empirical, prior_art, mechanism_hypothesis, gap, failure_mode }
  confidence: enum { load_bearing, supporting, exploratory }
  sources: list[SourceRef]  // canonical_id pointers; empty for unverifiable_by_design types
  failure_modes_if_wrong: string
  verification_status: enum { unverified, verified, partially_supported, contradicted,
                              source_not_found, unverifiable_by_design, verifier_error }
  round: int { 1, 2 }
  responding_to: list[anonymized_lens_id]  // Round 2 only, optional
}
```

**LensRun output** — the typed envelope:

```
LensRunOutput {
  findings: list[Finding]
  open_questions: list[string]
  disagreements_with_my_own_framing: list[string]  // MANDATORY, min 1
  brief_summary: string  // max 150 words
  refused: optional bool  // with reason if true
}
```

**Brief** — the 9 fields:

```
Brief {
  version: int
  problem_statement: string
  proposed_solution: optional string
  researcher_context: string
  background_claims: list[Finding]  // typed claims user is treating as established
  success_criteria_for_deliberation: string  // NEVER drafted by master
  scope_and_non_scope: string
  panel_constraints: optional list[lens_id]
  prior_panel_consultations: list[lens_run_ref]  // refinement only
  mode: enum { exploratory, deep_dive }
}
```

### API contracts between modules

- **IntakeOrchestrator → BriefVerificationStream:** stream of `background_claims` as they're added/edited during intake; receives stream of VerificationResults to surface at end of intake.
- **IntakeOrchestrator → PanelRouter:** confirmed Brief; receives proposed panel with reasoning.
- **PanelRouter → Round1Controller:** confirmed (Brief, panel) → DispatchEvent. Round1Controller spawns LensRuns.
- **Round1Controller → Round2Controller:** Round 1 LensRuns reaching terminal state → Round2Controller decides participants, builds anonymized inputs, dispatches.
- **LensRun → ClaimVerifier:** `verifier_query` tool call passes (claim_text, claim_type, sources); returns VerificationResult.

## Testing Decisions

### What makes a good test for this codebase

Tests must exercise external behavior, not implementation details. For this system specifically:

- **Lens output validity is testable; lens output quality is not** — tests confirm schema validity, mandatory-field enforcement, refused-as-legitimate-outcome. Quality is measured separately via eval (Layer 6, deferred).
- **Verifier behavior is testable end-to-end with fixtures** — given a (claim, mocked source DB response) pair, the verifier produces a deterministic VerificationResult. Fixtures cover: source-not-found, partially-supported, contradicted, verifier_error sub-reasons, passage-hallucinated downgrade.
- **Controllers are testable via simulated LensRun outcomes** — given fixtures of (valid / invalid / timeout / refused) lens responses, controllers reach correct terminal state, surface correct status, fire the >70% halt correctly.
- **State transitions on the data model are testable in isolation** — Brief version immutability, DispatchEvent distinguishing re-runs, Session status progression.

### Modules to be tested

All 6 deep modules get tests in v0, consistent with the user's clarification that v0 means "version I'd actually trust on real research."

- **IntakeOrchestrator** — scripted intake conversations covering mode declaration in both modes, field elicitation order, draft-after-turn-3 behavior, never-draft-success-criteria invariant, termination criteria per mode, the 3 decline gates, two-screen confirmation flow, brief version bump on edit-after-confirm.
- **BriefVerificationStream** — briefs with mixed verified / unverified / source-not-found / partially-supported claims, confirming correct tagging propagates to dispatch, confirming end-of-intake surfacing presents 3-option choice per problem, confirming silent background operation during intake conversation.
- **ClaimVerifier** — fixtures for each claim_type, each VerificationResult status, each failure sub-reason; passage-hallucination downgrade; cache hit/miss across (source, claim) combinations; cache invalidation on verifier_version bump.
- **PanelRouter** — briefs across modes and domains confirming routing rationale is per-brief-specific not boilerplate; `panel_constraints` overrides honored; exclusion reasoning produced for every excluded lens.
- **Round1Controller** — simulated lens responses (valid / invalid-once-then-valid / invalid-twice / timeout / no-progress-timeout / refused); >70% failure halt; parallel dispatch behavior; correct status surfaced to synthesis input.
- **Round2Controller** — Round 1 fixtures of varying sizes/shapes; correct Round 2 participant selection (mandatory + opt-in matching the `disagreements_with_my_own_framing`-references-peers heuristic); correct anonymization (lens identities literally absent from input); skip-condition logic (<3 successful Round 1 LensRuns); 70% Round 2 failure runs synthesis on Round 1 only.

### Prior art for tests in this codebase

No prior code in this repo — this is a greenfield v0. The user has built related infrastructure (claim verification, retrieval-augmented agents) and should mirror test patterns from those prior projects where applicable, especially for the ClaimVerifier module.

## Out of Scope

### Out of scope for this PRD (covered in a follow-up PRD)

This PRD covers Layers 1–3. The following are deferred to a follow-up PRD covering Layers 4–7:

- **Layer 4 — Human-in-the-loop:** challenge loop (user ↔ specific sub-agent re-engagement, anti-sycophancy guardrails, re-synthesis triggers), refinement loop (brief updates, partial re-runs, prior-output reuse-vs-invalidate logic), verdict loop (long-horizon feedback).
- **Layer 5 — Output & UX:** output structure for the researcher (synthesis-first vs per-agent-first, citation rendering, unverified claim rendering, visual surfacing of disagreements), persistence & session state.
- **Layer 6 — Quality & operations:** evaluation strategy (golden set, lens-ablation eval, self-use log), failure modes & system-level safeguards (silent agent failures, schema cascades, verifier rate limits, prompt-injection from retrieved papers, runaway loops), observability & debugging (logging, traces, replay).
- **Layer 7 — Explicit out-of-scope:** definitive list of what v0 deliberately doesn't do.

### Out of scope for v0 entirely (deferred to v1 or later)

- **Multi-user.** Single-user only. No auth, no permissions, no sharing.
- **Cost optimization.** System-side latency and cost are unconstrained.
- **Model diversity.** Single base model (Claude) across all lenses. Diversity comes from methodological frames, not from different LLMs.
- **Fine-tuned lenses.** All lenses are prompted, not fine-tuned.
- **Agent learning across sessions.** Lenses do not learn from prior sessions; statelessness is enforced per round.
- **Mobile.** Desktop / web only.
- **The cognitive science lens.** Cut from v0 due to woo risk. May return in v1 only if a real characteristic move can be specified that isn't covered by mech-interp or first-principles lens.
- **Synthesis (Round 3).** Decided to be in v0 architecturally, but the synthesis layer design is in the follow-up PRD.
- **Tool access beyond `verifier_query` and `source_fetch`.** No web search, no code execution, no arbitrary HTTP in lens tool access for v0.
- **Pre-dispatch harmful-content checks.** Single-user research tool; pre-filtering for harm is theater for this use case.
- **Pre-dispatch "out-of-research-domain predictions" decline gate.** Cut for single-user v0; would return if the tool ever goes multi-user.

## Further Notes

### Abandon triggers (forcing functions)

Three independent failure modes, each tied to a specific observable, each checkable on the first real run. The user has pre-committed to running the (d) check on the Active Context Inhibition proposal as the very first real action after v0 lands, before any feature polish:

- **(a)** >10% of cited papers/claims fail verification → credibility-laundering, kill.
- **(c1')** >60 min user-side attention for one deliberation cycle on a real problem → no time savings vs. substitutes, kill.
- **(d)** On the Active Context Inhibition proposal specifically, council surfaces fewer than 2 of the 5 gaps already identified during this design session → architecture wrong, kill.

The 5 gaps to check against on the (d) trigger are:
1. The Filter Agent has a circularity problem (if the filter is an LLM processing the long context, it has the same problem the system is trying to solve).
2. Du et al.'s "Retrieve-then-Solve" framed as proof-of-concept is also the existing solution space — the proposal's baselines exclude RAG and context-compression methods (LLMLingua, RECOMP, AutoCompressor).
3. The PFC analogy is rhetorical, not mechanistic — load-bearing in the framing but doesn't constrain the architecture.
4. Citation auditing risk — specific numerical claims and citations may not be quotable from the cited papers; some sources may be misattributed or not exist.
5. Agent vs. static long-context conflation — motivation is agentic (codebases, legal histories with dynamic context construction), but evaluation is static long-context QA with distractor text.

### Things to track during dogfooding

- The 5% manual verification audit for the first month (calibrates whether the structural hallucination detection is sufficient).
- Whether `disagreements_with_my_own_framing` mandatory minimum-1 produces real self-disagreements or fabricated ones (if fabricated, lens prompts need stronger guidance, possibly the field becomes optional with quality-eval pressure instead).
- Whether the 10-minute LensRun timeout is generous, tight, or correct (calibration unknown until first run).
- Whether the 70% Round 1 failure halt threshold is the right place (currently arbitrary, refine on data).
- Whether the first-principles lens's output-type restriction (mechanism_hypothesis + gap only) holds — if the lens consistently produces other claim types and gets rejected, the lens itself may not be viable in v0.
- One manual Verdict must be written against the Active Context Inhibition session within 90 days of running it, to exercise the Verdict schema before it drifts.

### Build sequence reminder

Verifier ships first → tier 1 lenses (prior-art + adversarial) tested on the Active Context Inhibition brief → if tier 1 output is trustworthy, tier 2 lenses ship in parallel. At deployment all 9 lenses are peers; the gating is build-order, not runtime priority.

### Open design questions deferred to the follow-up PRD

- How synthesis weights Round 1 vs Round 2 findings (the `round` and `responding_to` fields exist on Finding for this purpose, but synthesis logic is Layer 5).
- How challenge loop interacts with brief versioning (a challenge that produces new findings — does this bump brief version? Probably no, but the Challenge object's lifecycle needs spec.).
- What "materially different" means for brief versions in the refinement loop (which prior LensRuns are reusable vs. invalidated when a Brief edit happens).
- How the verdict loop captures researcher feedback (data model exists; UX/timing for capture is open).
