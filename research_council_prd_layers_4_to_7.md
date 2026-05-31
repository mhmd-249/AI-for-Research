# Research Council — PRD (Layers 4–7)

**Scope of this PRD:** human-in-the-loop (challenge / refinement / verdict loops), synthesis and output structure, persistence, quality and operations, and the explicit v0 out-of-scope edge. This is the follow-up to *Research Council — PRD (Layers 1–3)*, which covered foundations, infrastructure, and orchestration through the Round 2 controller. **Layers 1–3 are locked ground truth.** Every story below references those decisions by their locked behavior; nothing in Layers 1–3 is re-opened here. User-story numbering continues from the Layers 1–3 PRD (which ended at 101) so that cross-references by number remain globally unique.

---

## Problem Statement

The Layers 1–3 PRD gets a research brief through intake, dispatches it to a panel of methodological lenses in Round 1, and runs adversarial cross-pollination in Round 2 — all producing typed, per-claim-verified Findings. But Findings are not a deliverable. As the researcher, I still need three things the current scope doesn't provide:

1. **A way to fight back.** When a lens emits a Finding I think is wrong, I need to re-engage that lens specifically — and I need to be able to tell whether it *updated on my argument* or merely *caved to pressure*. A challenge loop that always makes the lens move is a yes-man with extra steps, which is the exact failure the whole anti-sycophancy commitment exists to prevent.
2. **A way to evolve the brief without throwing away good work.** Research framings shift mid-deliberation. I need to edit the brief and re-run, without nuking every lens's analysis over a typo and without silently carrying forward a Finding whose premise I just changed.
3. **Something I actually read.** Nine lenses' worth of typed Findings is not consumable in the 60-minute attention budget that is an abandon trigger. I need a synthesis that leads with what kills my proposal, encodes trust at a glance, and never averages genuine disagreement into mush. Without this, the tool is "impressive but unused" — the dominant failure mode to design against.

And underneath all three: I need to know the tool is *working* rather than *impressive*, to debug it when it isn't, and to have hard scope edges so I don't quietly re-expand v0 into a project I never ship.

## Solution

Four loops, a synthesis layer, an output contract, persistence, and an operations spine — all built as reads and narrow extensions over the typed data model already locked in Layers 1–3, not as new subsystems.

1. **Challenge loop (Layer 4.1):** I challenge a specific **Finding**, never a lens's whole analysis and never a synthesis judgment directly. The challenge produces a *scoped, single-Finding re-run* — a narrow new LensRun type that touches only the challenged claim, carries a mandatory `change_reason`, and may *flag* (never silently revise) sibling Findings it believes are now affected. Caving is *labeled*, not *blocked*. Re-synthesis recomputes on any accepted response.
2. **Refinement loop (Layer 4.2):** I edit the brief; the master diffs old vs. new and *proposes* an invalidation set with per-lens reasoning; I confirm or override. Reused LensRuns are re-pointed to the new brief version; invalidated ones re-dispatch. User-declares-scope, system-proposes — the same pattern as PanelRouter's Screen 2.
3. **Verdict loop (Layer 4.3):** Data model only in v0 — no UI, no behavior. A hand-authored, schema-validated Verdict file captures what actually happened, written in loose human references that a resolver lazily binds to precise Finding IDs.
4. **Synthesis (Layer 5.1 — "Round 3"):** A **hybrid** synthesizer. Deterministic code computes the structural layer (clusters, agreement counts, contradiction pairs) from typed Findings; a constrained LLM writes narration *over* that fixed structure and drafts conditional recommendations, with no authority to alter what code computed. Same-claim clustering is embedding-candidate grouping plus a boxed binary adjudicator that splits on uncertainty.
5. **Output (Layer 5.2):** Synthesis-first, gaps-and-tensions before agreements, inline scannable trust encoding, emergent (synthesizer-inferred) gaps in a distinct lower-trust band.
6. **Persistence (Layer 5.3):** A single embedded DB with a *global* Sources table; between-rounds-only resume off persisted state; no cross-session brief lineage.
7. **Quality & ops (Layer 6):** Eval is a read over the Verdict loop plus four hand-logged fields; cross-cutting safeguards are cheap additions to locked paths; observability is a backward walk over the persisted pointer graph plus an append-only raw-I/O trace table.

Two carried-over commitments shape everything: **system-side cost and latency are unconstrained, user-side attention is the scarce resource** (60 min/cycle is an abandon trigger), and **v0 means "the version I'd actually trust on real research," not a minimal demo.**

## User Stories

### Challenge loop (Layer 4.1)

102. As a researcher, I want to challenge a specific **Finding** rather than a whole LensRun or a synthesis judgment, so that I interrogate one coherent claim along the solution path the lens committed to, rather than letting the lens swap in true-but-off-path reasoning that loses the thread I was pulling on.
103. As a system, the `Challenge` object (stubbed in story 14) carries `challenged_finding_ref` (which transitively identifies its LensRun), `challenge_text`, and a reference to the produced challenge response, so that user↔lens re-engagement has first-class data representation.
104. As a researcher challenging a synthesis-level construct (an agreement cluster, a tension, or a surfaced gap), I cannot challenge it directly — I challenge the constituent Findings that rolled up into it, and synthesis is *recomputed*, not challenged, so that derived artifacts stay derived and I edit inputs rather than outputs.
105. As a challenge, I produce a **scoped single-Finding re-run**: a narrow new LensRun *type*, not the full Round 1/2 envelope, so that only the challenged claim can change and the lens is not re-run wholesale.
106. As a scoped challenge LensRun, I receive exactly `(brief, this lens's prior LensRuns on this brief, challenge_text, challenged_finding_ref)` — stateless reconstruction identical in spirit to the Round 1/2 contract — and I do NOT receive other lenses' work, because a challenge is not cross-pollination (Round 2 owns that).
107. As a scoped challenge LensRun, I emit a **stripped envelope**: one revised-or-reaffirmed Finding carrying a mandatory `change_reason`, plus optional sibling-impact flags; I do NOT emit a new `brief_summary` or a fresh full `disagreements_with_my_own_framing` list, so that the response is surgical rather than a whole-lens re-run in disguise.
108. As a challenge response Finding, I carry a `change_reason` enum — `unchanged`, `revised_new_evidence`, `revised_reconsidered`, `withdrawn`, `reaffirmed_against_challenge` — so that "the lens updated on evidence" is distinguishable from "the lens caved under pressure," which is the single field that makes the anti-sycophancy commitment auditable.
109. As a lens responding to a challenge, `reaffirmed_against_challenge` is a first-class, encouraged outcome — I am prompted that I may tell the researcher they are wrong and hold the Finding — so that the researcher debates a steelman rather than a yes-man.
110. As a challenge response that revises a Finding, I create a new Finding carrying a `supersedes` pointer to the challenged Finding; the old Finding stays immutable and persisted; the new Finding wins at synthesis render time.
111. As a system, superseded Findings are **stored but not shown** — hidden from the synthesis the researcher reads, but never deleted — so that the researcher is not made to re-live every revision the lens went through (which would make the tool harder than doing it themselves), while the Verdict schema, the 5% manual audit, and "historical session data recoverable as eval material" still have the data they depend on.
112. As a scoped challenge LensRun, I may *flag* (never silently edit) sibling Findings in my prior LensRun that I believe are undermined by the change to the challenged Finding, surfaced as an `open_questions`-style entry (e.g., "I withdrew F; this likely undermines my earlier point about G, but I was not asked to revise G"), so that an orphaned premise becomes a visible tension in synthesis rather than a silently stale Finding.
113. As a system, I do NOT carry an intra-LensRun `depends_on` field on Findings — sibling impact is prose-flagged, not structurally tracked — accepting that the researcher may need a second scoped challenge to revise a flagged sibling, in exchange for not burdening lens prompts and the schema with dependency declarations.
114. As a system, anti-sycophancy is **detect-and-label, not block**: a `revised_reconsidered` Finding (reconsidered on argument, no new source cited) is *visibly marked in synthesis* ("revised under challenge — no new evidence cited"), but never refused or gated, so that caving is legible at a glance without building a hard re-justification gate that is over-engineering for single-user v0.
115. As a system, a challenge does NOT bump brief version — the brief is unchanged; the challenge is dispatch-time activity on the existing brief — resolving the Layers 1–3 deferred question about challenge↔brief-versioning.
116. As a system, re-synthesis is triggered on any accepted challenge response, so that the synthesis the researcher reads always reflects the current winning set of Findings.

### Refinement loop (Layer 4.2)

117. As a researcher editing a confirmed brief, I want **user-declares-scope, system-proposes**: the master diffs old vs. new and proposes an invalidation set, and I confirm or override, so that I keep the master's "you forgot lens X also depended on this" catch without surrendering control to an opaque auto-diff classifier.
118. As a system, any brief edit produces a new brief version (immutability holds, per locked story 9/76); the edit never silently mutates a confirmed brief.
119. As the master proposing an invalidation set, I produce one-line reasoning per affected lens covering both re-run and reuse decisions (e.g., "you changed `proposed_solution`; prior-art, adversarial, architecture findings were load-bearing on the old solution → re-run; information-theoretic findings didn't reference it → reuse"), so that wrong calls are spottable — the same inclusion-AND-exclusion-reasoning pattern locked for PanelRouter Screen 2.
120. As a system, a brief edit is **material to a lens iff that lens had a `load_bearing` or `supporting` Finding whose `claim_text` referenced the edited content**; `exploratory` findings and untouched fields default to reuse — giving "materially different" (the Layers 1–3 deferred question) a concrete, mechanically-proposable definition with the user as backstop.
121. As a system, reused LensRuns are *re-pointed* to the new brief version with a `reused_from_version` marker — not re-executed — so that cross-run comparability is preserved and the unconstrained-cost budget is not spent re-running lenses an edit didn't touch.
122. As a system, invalidated LensRuns re-dispatch as new LensRuns against the new brief version.
123. As a system, I accept a known limitation: a reused Finding is "verified against brief v1, displayed under brief v2," and if the edit changed something the Finding *implicitly* assumed but did not cite, the material-edit heuristic (which checks only explicit `claim_text` references) will not catch it; the user-confirm step is the mitigation, not an airtight guarantee, and the 5% manual audit is the backstop if it bites.

### Verdict loop (Layer 4.3)

124. As a researcher, the Verdict loop is **data model only in v0** — no capture UI, no master-agent verdict flow, no loop behavior — so that the schema is exercised (locked story 20: one Verdict against the ECF session within 90 days) without building capture UX I explicitly don't want.
125. As a researcher writing a Verdict, I author a structured file (JSON/YAML matching the schema) that is validated against the schema on load, so that I — the only author and an engineer — can exercise the schema by hand.
126. As a Verdict, my `predictions_validated` and `predictions_invalidated` are written as **loose human references** `(lens, claim_text excerpt)`, so that I focus on the scientific material and write what is natural, not on learning the system's internal IDs months after the session.
127. As a system, each loose Verdict reference carries a **derived, cached `resolved_finding_ids` field** that a resolver populates lazily (fuzzy-matching the excerpt to a Finding within the session), so that eval mining later reads precise Finding-granular pointers while the researcher only ever touched the excerpt — write loose, system resolves to precise, keep both.
128. As a Verdict, I carry a `decision` enum — `pursued_as_proposed`, `pursued_modified`, `abandoned`, `superseded_by_other_work`, `undetermined` — plus an `outcome_summary` free-text field, so that the calibration question that justifies the whole tool ("on sessions where I abandoned the approach, did the council surface the killing gap before I committed?") aggregates rather than living only in un-queryable prose.
129. As a Verdict, my prediction pointers resolve to **specific Findings**, not lenses or synthesis clusters, so that the eval question "when lens X emitted Finding Y at `load_bearing` confidence, was it right?" is answerable at the granularity calibration actually needs.

### Synthesis layer — "Round 3" (Layer 5.1)

130. As a system, the synthesizer is **hybrid**: deterministic code computes the structural layer, and a constrained LLM writes narration over that fixed structure — never pure-LLM (which would reintroduce prose-on-typed-input and let the synthesizer hallucinate a consensus the Findings don't support, the exact failure the typed-Finding architecture exists to prevent).
131. As the deterministic layer, I compute agreement clusters, agreement counts, tension/contradiction pairs, and gap collections directly from typed Finding fields, so that "agreement is structural, computed at synthesis, displayed as counts" (locked stories 27/28) is honored by construction.
132. As the LLM narration pass, I am handed the computed clusters as fixed input, I write readable prose over them, and I draft the conditional recommendations — but I cannot alter cluster membership or counts, enforced by re-validating my output against the computed structure (same spirit as the locked two-strike, code-enforced schema check).
133. As the same-claim clustering step, I use **embedding-similarity candidate grouping plus a boxed binary LLM adjudication pass** that answers only "are these two Findings asserting the same thing? y/n" per candidate pair — never free-form clustering — so that the one irreducibly-semantic judgment in synthesis is auditable and fixture-testable.
134. As the adjudicator, when I am genuinely uncertain whether two Findings are the same claim, I **default to split** (treat them as distinct), so that anti-averaging is preserved and the error I make is the *visible* one (two adjacent clusters the researcher can merge by eye) rather than the *invisible* one (a false "7 lenses agree" that silently misleads).
135. As the synthesizer, I display cross-lens agreement as counts with confidence breakdown (e.g., "6 of 9 lenses independently flagged this; 3 load_bearing, 3 supporting"), never as a numerical confidence percentage (locked story 28).
136. As the synthesizer, I surface gaps in **two tiers**: *declared gaps* (clustered `gap`-type Findings, deterministic, verified against the brief per locked story 25) and *emergent gaps* (tensions between lenses' Findings that imply an unaddressed problem, named by the narration pass, clearly labeled synthesizer-inferred and lower-trust), so that the researcher can distrust connect-the-dots gaps appropriately while still benefiting from them — and so that the ECF abandon-trigger (surface ≥2 of 5 known gaps) has a mechanism that can catch emergent gaps no single lens emitted.
137. As the synthesizer, I detect contradictions structurally via `contradicted` verification status and opposing `failure_mode`/`mechanism_hypothesis` pairs, and I preserve a split (e.g., 6 vs. 4) as two clusters with counts, never as "mostly agree" — anti-averaging by construction.
138. As the `Synthesis` object, I **reference** (not copy) Findings (locked story 15), so that challenge-driven supersessions and refinement-driven invalidations propagate into the synthesis the researcher reads without a stale copy.

### Output structure for the researcher (Layer 5.2)

139. As a researcher, I read **synthesis-first**: the agreement map / tension map / gap list / conditional recommendations are the primary artifact, and per-lens raw output is *drill-down*, reached by expanding a synthesis claim to see its constituent Findings and which lenses emitted them — so that I am not handed nine essays I read once and never reopen (the "impressive but unused" trap).
140. As a researcher, synthesis is ordered **gaps and tensions first, agreements last**, so that for a gap-finding tool the load-bearing output (what's wrong with my proposal) leads and the reassurance (what the council agrees on) trails — I lead with the thing that can change my mind.
141. As a researcher, within the gaps-and-tensions band, items are **severity-ordered**, and `exploratory`-confidence gaps with no cross-lens support drop into a collapsed "minor / speculative" subsection, so that a healthy proposal opens to a short gaps section with weak objections folded away — gaps-first that builds trust rather than gaps-first that trains me to ignore it (the cry-wolf failure).
142. As a researcher, every rendered claim carries **inline, scannable, fixed-position trust encoding**: its cross-lens count, the strongest confidence tier in its cluster, and its verification-status marker (`unverified` / `contradicted` / `unverifiable_by_design` surfaced honestly, never hidden), so that a `contradicted` Finding that several lenses leaned on is *visually alarming* rather than a footnote.
143. As a researcher, emergent (synthesizer-inferred) gaps render in a **distinct, explicitly-lower-trust band** ("the synthesizer noticed this tension implies an unaddressed problem; no lens asserted it directly"), so that I can tell at a glance whether a gap came from a lens (verified against the brief) or from the synthesizer connecting dots (inferred).
144. As a system, this layer specifies output *structure* — ordering, hierarchy, and what metadata attaches to each claim — format-agnostic; UI design is out of scope (locked).

### Persistence & session state (Layer 5.3)

145. As a system, all state persists to a **single embedded database** (SQLite or equivalent) with a **global, cross-session `Sources` table**, so that the Source cache (locked story 13: dedup and reuse across Findings, LensRuns, and Sessions) is genuinely global rather than fragmented across per-session files.
146. As a researcher, a Session is resumable **between rounds only**: once a deliberation round is dispatched it runs to terminal completion before the Session can pause, and resume reloads the Session from persisted state at a round boundary — so that the state machine stays dead simple, with no mid-round persistence, partial-LLM-call restoration, or re-dispatch idempotency burden.
147. As a system, the Session status machine is `intake` / `consulting` / `synthesized` / `iterating` / `closed` (locked story 8), and between-rounds resume aligns with the Round 1 controller's existing terminal-state definition (locked story 93), so that the poll-able boundary the controller already computes *is* the resume boundary — making this resume model essentially free.
148. As a system, there is **no cross-session brief lineage** in v0: a Session owns its Briefs, `prior_panel_consultations` (locked story 69) handles within-session refinement pointers, and cross-session continuity ("ECF-v2 evolved from last month's ECF session") is achieved by starting a new Session and manually copying prior brief content — deferring any Project/Topic-above-Session object to v1.

### Evaluation strategy (Layer 6.1)

149. As a researcher, the **golden set** begins with the ECF proposal as entry #1, with its 5 known gaps (enumerated in the Layers 1–3 abandon-triggers section) as the labeled answer key, so that the first eval is the first real run.
150. As a researcher, the golden set grows by **one labeled case per Verdict-producing dogfooding session** — the Verdict's `predictions_validated`/`predictions_invalidated` *are* the labels — so that a single-user tool, which cannot crowdsource labels, uses dogfooding itself as the labeling pipeline (the golden set and the Verdict loop are the same thing viewed twice).
151. As a researcher, **lens-ablation eval is manual and rare**, not automated: I re-run a golden-set brief with lens X removed and check whether the gaps lens X was responsible for disappear from synthesis (the cognitive-science-lens-cut logic generalized), supported by the locked DispatchEvent model (stories 17/85) that distinguishes panels on the same brief — automating it is Layer 6 over-build.
152. As a researcher, v0 tracks exactly two metrics: **verification-fail-rate** as the *guardrail* metric (it is the locked abandon trigger) and **gap-recall on Verdict-labeled sessions** as the *value* metric (of the gaps I later confirmed real via Verdict, what fraction did the council surface before I committed) — the entire value proposition as one number, fed for free by the Verdict loop.
153. As a researcher, I keep a **four-field self-use log**, hand-entered at session close — time-to-dispatch, total user-attention-minutes, did-I-act-on-the-output, did-I-trust-it — as the *leading* indicator of "impressive but unused," before enough Verdicts accumulate to compute the *lagging* gap-recall metric.
154. As a researcher, the self-use log is **soft-honest-self-report**, accepting that I am builder, rater, and beneficiary and the signal is gameable by motivated optimism; this is defensible because v0 is single-user (there is no one to deceive but myself) and the lagging gap-recall metric is the backstop that eventually catches a fudged self-rating.
155. As a researcher, as a dogfooding discipline (instruction, not enforced mechanism) I log the "did-I-trust-it" field *before* reading the synthesis quality, so that the rating is not retrofitted to justify effort already spent.

### Failure modes & system-level safeguards (Layer 6.2)

156. As a system, a lens that returns valid schema and is not `refused` but emits **zero findings** is treated as a *soft anomaly surfaced in synthesis* ("Lens X succeeded but produced no findings — possible silent failure"), not a hard error — a single render rule catching the silent-agent-failure seam that per-component handling (locked stories 45/46/53) doesn't cover.
157. As a system, before clustering, a thin **pre-clustering sanity filter** (non-empty `claim_text`, minimum length) routes schema-valid-but-degenerate Findings to the same "excluded with note" path that schema-invalid Findings use (locked stories 50/51), so that the clustering step (story 133) cannot choke on whitespace-only or empty claims.
158. As a system, the case where all parallel Round 1 lenses (locked story 87) hit the verifier simultaneously and the verifier's internal queue backs up past the 10-minute LensRun timeout (locked story 89) is **deliberately NOT hardened**: backed-up verifier → some claims stay `unverified` → surfaced tagged in synthesis (locked stories 37/39), which the existing timeout + verifier-error-tagging machinery composes correctly — marked considered-and-accepted so it is not an accidental gap.
159. As a system, retrieved source text (pulled by `source_fetch` lenses — prior-art, adversarial, empirical-benchmarking) enters lens context only inside a **delimited, clearly-labeled "untrusted retrieved content" quarantine block**, with the lens prompt stating retrieved text is data to analyze, never instructions to follow, so that prompt-injection from a poisoned abstract is contained — a different threat from the brief-content harm checks correctly cut in Layers 1–3.
160. As a system, the verifier — not the lens — assigns `verification_status`, and the locked passage-hallucination string-match (story 40) means a lens cannot fabricate a verdict even if a poisoned paper convinces it to *claim* "verified"; the injection's blast radius is therefore capped at *what the lens writes in its findings*, not the verification verdict. No separate sanitization pass is built (it is harm-theater for a single-user tool with no targeted adversary; the realistic threat is an accidentally weird abstract).
161. As a system, runaway iteration is bounded by a **per-Session hard cap of 10 challenge+refinement cycles, surfaced as a signal rather than silently enforced** ("you've run 10 refinement cycles on this Session — the council may be telling you the brief itself is the problem"), so that the cap doubles as a diagnostic that the problem is upstream of the tool.

### Observability & debugging (Layer 6.3)

162. As a researcher debugging "why did the council say this," I get a **trace-view that walks the persisted pointer graph backward** for any synthesis claim: synthesis claim → constituent Findings → emitting LensRuns → brief version → VerificationResult → Source text — a read over data the over-built data model already stores, not a separate logging system.
163. As a system, I capture **raw `(prompt, completion, model, timestamp)` per LLM call in a separate append-only trace table**, keyed to the LensRun or verifier stage that made the call — kept out of the domain objects to keep the clean data model clean — because "why did lens X emit this garbage" is unanswerable from typed Findings alone (statelessness means the prompt is reconstructable but not otherwise stored).
164. As a system, the raw-I/O trace table is **unbounded for v0**: typed Findings are the permanent eval record, raw I/O is the permanent debug record, disk is cheap, and retention logic is premature optimization for single-user.
165. As a researcher, **replay is partial only**: I can manually re-dispatch a single LensRun or a single verifier check against a stored brief version (the same stateless reconstruction the system does normally, triggered manually for debugging) — full-session deterministic replay is not achievable (LLM nondeterminism) and not faked.

### Explicit out-of-scope (Layer 7)

166. As a system, I do NOT support **synthesis-level direct challenge** — the researcher challenges constituent Findings and synthesis recomputes (from story 104).
167. As a system, I do NOT **re-run a whole lens on challenge** — scoped single-Finding only; sibling ripple is flagged, never auto-revised (from stories 105/112).
168. As a system, I do NOT build **hard anti-sycophancy gates** — caving is labeled, never blocked (from story 114).
169. As a system, I do NOT carry an **intra-LensRun dependency graph** (`depends_on` on Findings) — sibling impact is prose-flagged (from story 113).
170. As a system, I do NOT build an **automated lens-ablation harness** — ablation is manual and rare (from story 151).
171. As a system, I do NOT build a **Verdict capture UI or master-agent verdict flow** — hand-authored validated file only (from story 124).
172. As a system, I do NOT build **cross-session brief lineage or a Project/Topic-above-Session object** — manual copy for continuity (from story 148).
173. As a system, I do NOT support **mid-round pause/resume** — between-rounds-only (from story 146).
174. As a system, I do NOT build **full-session deterministic replay** — partial re-dispatch only (from story 165).
175. As a system, I do NOT build a **retrieved-content sanitization pass** — quarantine block + verifier-owns-the-verdict only (from stories 159/160).
176. As a system, I do NOT build a **behavioral/objective self-use signal** — soft-honest-self-report only (from story 154).
177. As a system, I do NOT build a **pure-LLM synthesizer** — hybrid only, LLM cannot alter computed structure (from stories 130/132).
178. As a system, I do NOT impose **trace-table retention bounds** — unbounded for v0 (from story 164).
179. As a system, I additionally and explicitly exclude from v0: **notifications/scheduling** (the council does not run unattended and ping me), **export/sharing of synthesis reports** (single-user; no PDF/collaborator export), a **cost/token dashboard** (cost is unconstrained *and* not surfaced — unconstrained ≠ visible), and **multi-brief / batch dispatch** (the DispatchEvent model supports running one brief through multiple panels and diffing them, but v0 does not provide a batch workflow for it).
180. As a system, this Layer 7 list is **stacked on top of — not replacing — the Layers 1–3 v0 out-of-scope list** (multi-user, cost optimization, model diversity, fine-tuned lenses, cross-session agent learning, mobile, the cognitive-science lens, tool access beyond `verifier_query`/`source_fetch`, pre-dispatch harmful-content checks, the out-of-research-domain decline gate).

## Implementation Decisions

### Domain glossary additions

Extends the Layers 1–3 glossary; existing terms keep their locked meanings.

- **Challenge** — user↔lens re-engagement targeting a specific **Finding** (no longer a stub; story 14 is now fully specified). Captures `challenged_finding_ref`, `challenge_text`, and the produced challenge response.
- **Scoped challenge LensRun** — a narrow LensRun *type* produced by a challenge: input includes `challenge_text` + `challenged_finding_ref`, output is a stripped envelope (one Finding + `change_reason` + optional sibling-impact flags), not the full Round 1/2 `LensRunOutput`.
- **`change_reason`** — enum on a challenge response Finding: `unchanged` / `revised_new_evidence` / `revised_reconsidered` / `withdrawn` / `reaffirmed_against_challenge`. The auditability hook for anti-sycophancy.
- **`supersedes`** — pointer on a revised Finding to the (immutable, hidden-but-persisted) Finding it replaces.
- **`reused_from_version`** — marker on a LensRun re-pointed to a new brief version during refinement instead of being re-executed.
- **Refinement** — brief edit producing a new brief version, with a system-proposed / user-confirmed invalidation set.
- **Verdict** — now fully specified (was data-modeled-only intent in story 16): hand-authored, schema-validated, loose `(lens, excerpt)` references with lazily-resolved cached Finding-ID pointers, `decision` enum, `outcome_summary`.
- **Synthesis (Round 3)** — the hybrid synthesizer: deterministic structure + non-authoritative LLM narration. Architecturally Round 3, locked in-scope for v0 in Layers 1–3, designed here.
- **Same-claim adjudication** — the boxed binary "are these two Findings the same claim?" judgment; the one place an LLM touches synthesis structure, split-on-uncertainty.
- **Declared gap / emergent gap** — gap clustered from `gap`-type Findings (deterministic) vs. gap named by the narration pass from inter-lens tension (synthesizer-inferred, lower-trust band).
- **Self-use log** — four hand-entered fields per session close; the leading indicator of "impressive but unused."
- **Trace record** — append-only raw `(prompt, completion, model, timestamp)` per LLM call, separate from domain objects.

### Module sketch (deep modules to build/test in isolation)

Continues the Layers 1–3 module list (1–6: IntakeOrchestrator, BriefVerificationStream, ClaimVerifier, PanelRouter, Round1Controller, Round2Controller). New deep modules:

7. **ChallengeOrchestrator** — owns the scoped single-Finding challenge: constructs the narrow challenge LensRun input `(brief, prior LensRuns, challenge_text, challenged_finding_ref)`, captures `change_reason`, manages the `supersedes` chain (new Finding wins, old hidden-but-persisted), routes sibling-impact flags into synthesis as tensions, and fires the re-synthesis trigger. Does NOT bump brief version.
8. **RefinementOrchestrator** — owns brief-edit → new brief version; diffs old vs. new; proposes an invalidation set with per-lens (re-run AND reuse) reasoning using the material-edit rule; takes user confirm/override; re-points reused LensRuns with `reused_from_version`; re-dispatches invalidated LensRuns.
9. **Synthesizer (Round3Controller)** — owns the hybrid pipeline: deterministic structure computation (agreement/tension/contradiction/gap maps from typed Findings), drives ClaimClusterer, runs the constrained LLM narration pass with output re-validation against computed structure, two-tier gap surfacing, and the output ordering rules (gaps-first, severity-ordered, weak-gaps-collapsed, emergent-gap band). Produces/updates the referencing `Synthesis` object.
10. **ClaimClusterer** — isolated deep module (the load-bearing technical risk of synthesis): embedding-similarity candidate grouping + boxed binary same-claim adjudication + split-on-uncertainty. Isolated specifically so it is fixture-testable, because if same-claim detection is wrong every count is wrong.
11. **SessionStore** — single embedded DB; global `Sources` table; persistence of all domain objects plus `supersedes` / `reused_from_version` markers; Session status transitions; between-rounds resume off persisted state.
12. **TraceStore** — append-only raw-I/O trace table (unbounded); the backward-pointer-walk trace-view for "why did the council say this"; partial replay via manual stateless re-dispatch.
13. **VerdictResolver** — validates hand-authored Verdict files against the schema on load; lazily resolves loose `(lens, excerpt)` references to cached `resolved_finding_ids` via fuzzy match within the session.
14. **EvalReader** — read over Verdict + DispatchEvent data; computes verification-fail-rate (guardrail) and gap-recall-on-Verdict-labeled-sessions (value); supports manual lens-ablation comparison across DispatchEvents on the same brief.

Cross-cutting data-model additions (objects/fields, not modules): `Challenge` (fully specified), scoped-challenge-LensRun type, `change_reason` enum, `supersedes` pointer on Finding, `reused_from_version` marker on LensRun, `Verdict` (fully specified: loose references + `resolved_finding_ids` cache + `decision` enum + `outcome_summary`), `Synthesis` (fully specified: referencing agreement/tension/gap/recommendation structure), self-use log record, trace record.

### Key architectural commitments (Layers 4–7)

- **A challenge edits inputs, never outputs.** Synthesis is derived; the researcher challenges constituent Findings and synthesis recomputes. Superseded Findings are hidden-but-persisted, never deleted.
- **Detect-and-label sycophancy; do not block it.** `change_reason` makes caving auditable; `revised_reconsidered`-no-source is visibly marked; no hard gate. Single-user v0 reads the tag itself.
- **Refinement reuses by re-pointing, not re-running.** Material-edit = touched a `load_bearing`/`supporting` Finding's referenced content. System proposes, user confirms — the PanelRouter Screen 2 pattern.
- **Synthesis structure is code; synthesis prose is LLM.** The deterministic layer is authoritative for clusters and counts; the LLM narrates over fixed structure and cannot alter it. Anti-averaging falls out by construction. Split-on-uncertainty makes the visible error, not the invisible one.
- **The data model already is the observability system.** "Why did the council say this" is a backward walk over persisted pointers. The only added capture is raw LLM I/O, in a separate append-only table, unbounded.
- **Eval is the Verdict loop plus four hand-logged fields.** No separate eval subsystem. Guardrail metric (verification-fail-rate) + value metric (gap-recall) are reads over existing data.
- **State persists; resume is between-rounds-only.** Single embedded DB, global Sources table. The dead-simple state machine is bought by the locked statelessness and the controllers' existing terminal-state boundary.

### Schemas (decision-encoding snippets)

**Challenge** — now fully specified (story 14 stub resolved):

```
Challenge {
  challenged_finding_ref: finding_id      // transitively identifies the LensRun
  challenge_text: string
  response_ref: scoped_challenge_lensrun_id
  bumps_brief_version: false              // invariant, never true
}
```

**Scoped challenge LensRun** — the narrow envelope (NOT a full LensRunOutput):

```
ScopedChallengeOutput {
  finding: Finding                        // single revised-or-reaffirmed Finding
  change_reason: enum { unchanged, revised_new_evidence, revised_reconsidered,
                        withdrawn, reaffirmed_against_challenge }
  sibling_impact_flags: list[string]      // prose flags; surfaced as tensions, never auto-edits
  // NO brief_summary, NO fresh disagreements_with_my_own_framing
}
```

**Finding addition** — supersession (extends the locked Layers 1–3 Finding):

```
Finding {
  ...locked fields (claim_text, claim_type, confidence, sources,
     failure_modes_if_wrong, verification_status, round, responding_to)...
  supersedes: optional finding_id         // set on a challenge revision; old Finding hidden-but-persisted
}
```

**Verdict** — fully specified:

```
Verdict {
  session_ref: session_id
  decision: enum { pursued_as_proposed, pursued_modified, abandoned,
                   superseded_by_other_work, undetermined }
  outcome_summary: string
  predictions_validated: list[VerdictRef]
  predictions_invalidated: list[VerdictRef]
}

VerdictRef {
  lens: lens_id
  claim_text_excerpt: string              // human-written, loose
  resolved_finding_ids: list[finding_id]  // derived/cached, populated lazily by VerdictResolver
}
```

**Synthesis** — fully specified, referencing (not copying) Findings:

```
Synthesis {
  agreement_clusters: list[{ finding_refs: list[finding_id],
                             count: int,
                             confidence_breakdown: { load_bearing: int, supporting: int, exploratory: int } }]
  tensions: list[{ finding_refs: list[finding_id], description: string,
                   source: enum { contradiction, sibling_impact_flag, opposing_pair } }]
  declared_gaps: list[{ finding_refs: list[finding_id] }]        // clustered gap-type Findings
  emergent_gaps: list[{ description: string, implicated_finding_refs: list[finding_id] }]  // synthesizer-inferred, lower-trust band
  conditional_recommendations: list[string]                      // LLM-drafted over fixed structure
  // ordering: gaps+tensions first (severity-ordered, weak gaps collapsed), agreements last
  // every rendered claim carries inline: cross-lens count + strongest confidence tier + verification marker
  // revised_reconsidered-no-source Findings carry a visible "revised under challenge — no new evidence" mark
}
```

**Self-use log** — four hand-entered fields per session close:

```
SelfUseLog {
  session_ref: session_id
  time_to_dispatch_minutes: number
  total_user_attention_minutes: number    // the 60-min abandon-trigger metric
  acted_on_output: bool                    // soft-honest-self-report
  trusted_output: bool                     // logged BEFORE reading synthesis quality (discipline)
}
```

**Trace record** — append-only, separate from domain objects:

```
TraceRecord {
  caller_ref: lensrun_id | verifier_stage_id
  prompt: string
  completion: string
  model: string
  timestamp: datetime
}
```

### API contracts between modules

- **ChallengeOrchestrator → Synthesizer:** accepted challenge response (winning Finding via `supersedes`, sibling-impact flags) → triggers re-synthesis; sibling flags enter as `tensions`.
- **RefinementOrchestrator → PanelRouter/Round1Controller:** new brief version + confirmed invalidation set → reused LensRuns re-pointed (`reused_from_version`), invalidated LensRuns re-dispatched as new DispatchEvent activity on the new version.
- **Round2Controller → Synthesizer:** terminal Round 1 (+ Round 2, if run) LensRuns → Synthesizer computes structure. (Round 2 skip/failure notes from locked stories 100/101 are carried into the Synthesis output.)
- **Synthesizer → ClaimClusterer:** typed Findings → candidate groups + per-pair same-claim verdicts → fixed cluster structure (split-on-uncertainty).
- **Synthesizer → LLM narration pass:** fixed cluster/tension/gap structure → prose + conditional recommendations, re-validated against the input structure (membership/counts unchangeable).
- **VerdictResolver → EvalReader:** resolved `resolved_finding_ids` per VerdictRef → gap-recall computation.
- **All LLM-calling modules → TraceStore:** raw `(prompt, completion, model, timestamp)` append on every call.
- **All modules → SessionStore:** persistence of domain objects + `supersedes`/`reused_from_version` markers; between-rounds resume reads.

## Testing Decisions

### What makes a good test for this codebase

Same principle as Layers 1–3: tests exercise external behavior, not implementation details. Quality remains eval-measured (Layer 6.1), not unit-tested. Specifically for Layers 4–7:

- **Loop behavior is testable via simulated lens responses and fixture briefs** — given a challenged Finding and a scripted challenge response, the ChallengeOrchestrator produces the correct `supersedes` chain, surfaces sibling flags as tensions, and triggers re-synthesis; given a brief diff, the RefinementOrchestrator proposes the correct invalidation set under the material-edit rule.
- **Clustering is testable with fixture Finding pairs** — given two Findings, ClaimClusterer's adjudicator returns deterministic merge/split, and the split-on-uncertainty default is exercised with deliberately-ambiguous pairs. This is the highest-priority test target: every synthesis count depends on it.
- **Synthesis structure is testable independently of narration** — given a fixed set of typed Findings, the deterministic layer produces correct counts, contradiction pairs, and two-tier gap collections; a separate test confirms the narration pass cannot alter membership/counts (re-validation rejects an LLM output that does).
- **Verdict resolution is testable with fixtures** — loose `(lens, excerpt)` references resolve to the correct Finding IDs; ambiguous/no-match excerpts are handled explicitly.
- **Persistence and resume are testable as state transitions** — Session status progression, global Sources dedup across sessions, between-rounds resume from persisted state, brief-version immutability under refinement.
- **Observability is testable as a graph walk** — given a persisted session, the trace-view reconstructs the correct backward chain for a synthesis claim; partial replay re-dispatches the correct single LensRun against the correct brief version.

### Modules to be tested

All 8 new deep modules (7–14) get tests in v0, consistent with the locked "v0 means the version I'd actually trust on real research" stance (carried over from Layers 1–3).

- **ChallengeOrchestrator** — scoped-single-Finding isolation (only the challenged Finding changes); `change_reason` capture including `reaffirmed_against_challenge`; `supersedes` chain with old Finding hidden-but-persisted; sibling-impact flag → tension surfacing; no-brief-version-bump invariant; re-synthesis trigger.
- **RefinementOrchestrator** — material-edit rule (touched `load_bearing`/`supporting` referenced content → re-run; `exploratory`/untouched → reuse); per-lens reasoning produced for both re-run and reuse; new brief version on edit; `reused_from_version` re-pointing vs. re-dispatch; user override of the proposed set.
- **Synthesizer** — deterministic counts/contradictions/gaps from fixture Findings; narration re-validation rejecting structure-altering output; two-tier gap separation; ordering rules (gaps-first, severity, weak-gaps-collapsed); `revised_reconsidered`-no-source visible mark; referencing (not copying) Findings so supersession propagates.
- **ClaimClusterer** — fixture Finding pairs for merge/split; split-on-uncertainty default; embedding-candidate grouping behavior; resistance to over-merging two distinct same-sounding gaps.
- **SessionStore** — global Sources dedup across sessions; Session status machine; between-rounds resume; brief-version immutability; `supersedes`/`reused_from_version` persistence.
- **TraceStore** — append-only raw-I/O capture per call; backward trace-view reconstruction; partial replay re-dispatch correctness; unbounded growth (no retention logic to test).
- **VerdictResolver** — schema validation of hand-authored files; loose-reference → Finding-ID resolution including ambiguous/no-match cases.
- **EvalReader** — verification-fail-rate and gap-recall computation over Verdict + DispatchEvent fixtures; lens-ablation comparison across DispatchEvents on the same brief.

### Prior art for tests in this codebase

The Layers 1–3 modules and their tests are now prior art for this codebase. ClaimClusterer should mirror the ClaimVerifier's fixture-driven, deterministic-given-mocked-inputs test pattern (locked Layers 1–3 testing decision), since both turn an irreducibly-semantic judgment into a boxed, auditable call. The Synthesizer's structure-vs-narration split mirrors the locked "schema validity enforced in code, quality enforced in prompt with eval as the verifier" division. The user's own prior infrastructure (claim verification, retrieval-augmented agents) remains the external reference for the ClaimVerifier-adjacent pieces.

## Out of Scope

### Out of scope for this PRD

Nothing further is deferred to a later PRD — Layers 4–7 complete the v0 design. Round 3 synthesis, which Layers 1–3 deferred to "the follow-up PRD," is designed here (Layer 5.1). The Layers 1–3 open design questions are resolved as follows: challenge↔brief-versioning (story 115: no bump); "materially different" for brief versions (story 120: touched a `load_bearing`/`supporting` Finding's referenced content); synthesis weighting of Round 1 vs. Round 2 findings (handled structurally — the locked `round`/`responding_to` fields feed clustering and tension detection; v0 does not apply a numerical weight, consistent with counts-not-percentages).

### Out of scope for v0 entirely

The full Layer 7 list (stories 166–180), stacked on the Layers 1–3 v0 out-of-scope list (story 180). In brief: no synthesis-level direct challenge, no whole-lens re-run on challenge, no hard anti-sycophancy gates, no intra-LensRun dependency graph, no automated lens-ablation harness, no Verdict capture UI, no cross-session brief lineage, no mid-round pause/resume, no full-session deterministic replay, no retrieved-content sanitization pass, no behavioral self-use signal, no pure-LLM synthesizer, no trace-table retention bounds, no notifications/scheduling, no export/sharing, no cost/token dashboard, no multi-brief/batch dispatch — plus everything already cut in Layers 1–3.

## Further Notes

### How Layers 4–7 close the abandon triggers

- **(a) >10% verification fail → kill.** EvalReader computes verification-fail-rate as the guardrail metric (story 152). The Layer 6.2 quarantine + verifier-owns-the-verdict (stories 159/160) protect this number from prompt-injection laundering.
- **(c1') >60 min user-side attention → kill.** The self-use log's `total_user_attention_minutes` (story 153) is the direct measurement. Synthesis-first + gaps-first + weak-gaps-collapsed (stories 139–141) are the design moves that defend the budget on the read side; scoped single-Finding challenges (story 105) defend it on the iterate side.
- **(d) <2 of 5 ECF gaps surfaced → kill.** Two-tier gap surfacing (story 136), especially emergent gaps, is the mechanism that can catch the ECF gaps no single lens emits (notably gap #1, the Filter Agent circularity, which lives in the tension between lenses). The golden set (story 149) makes this a repeatable check, not just a one-time first-run gate.

### Things to track during dogfooding (additions to the Layers 1–3 list)

- Whether **split-on-uncertainty** under-counts agreement enough to be annoying in practice (story 134) — if "3 lenses + 4 lenses" for an obvious single cluster of 7 happens often, revisit the adjudicator threshold rather than the default.
- Whether **emergent-gap** narration produces real connect-the-dots gaps or plausible-sounding noise (story 136) — if noise, tighten the narration prompt or demote emergent gaps further; this is the riskiest LLM-judgment surface in synthesis.
- Whether **`change_reason`** is filled honestly by lenses or gamed (story 108) — if lenses systematically avoid `reaffirmed_against_challenge` (always appearing to update), the steelman behavior isn't working and the lens prompts need stronger anti-sycophancy guidance.
- Whether the **material-edit heuristic** misses implicit-assumption staleness often enough to matter (story 123) — the "verified against v1, shown under v2" gap is the thing the 5% audit should specifically watch in refined sessions.
- Whether the **10-cycle challenge+refinement cap** is the right number (story 161) — currently arbitrary, refine on data; treat hitting it as a signal about the brief, not a nuisance limit.
- Whether **soft-honest-self-report** survives builder bias (story 154) — if `acted_on_output` and `trusted_output` stay implausibly high while gap-recall comes in low, the self-report is being fudged and the lagging metric is the truth.

### Build sequence reminder (Layers 4–7)

Synthesis (Layer 5.1) is the critical path — it is what turns the locked Round 1/2 infrastructure into something readable, and the ECF abandon-trigger (d) cannot be evaluated without it. Build order: **Synthesizer + ClaimClusterer first** (ClaimClusterer tested hardest, since every count depends on it) → **SessionStore + TraceStore** (needed to persist and debug everything else) → **ChallengeOrchestrator + RefinementOrchestrator** (the iterate loops, which depend on a working synthesis to iterate against) → **VerdictResolver + EvalReader** (the longest-horizon pieces; Verdict has no UI and EvalReader has little to read until Verdicts and self-use logs accumulate). The ECF run remains the first real action after the synthesis path is trustworthy, before any iterate-loop polish.
