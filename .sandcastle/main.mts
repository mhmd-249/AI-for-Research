// Parallel Planner with Review — four-phase orchestration loop
//
// This template drives a multi-phase workflow:
//   Phase 1 (Plan):             An opus agent analyzes open issues, builds a
//                               dependency graph, and outputs a <plan> JSON
//                               listing unblocked issues with branch names.
//   Phase 2 (Execute + Review): For each issue, a sandbox is created via
//                               createSandbox(). The implementer runs first
//                               (100 iterations). If it produces commits, a
//                               reviewer runs in the same sandbox on the same
//                               branch (1 iteration). All issue pipelines run
//                               concurrently via Promise.allSettled().
//   Phase 3 (Merge):            A single agent merges every branch that carries
//                               unmerged commits for a still-open issue — including
//                               branches stranded by an earlier iteration — into
//                               the current branch.
//
// The outer loop repeats up to MAX_ITERATIONS times so that newly unblocked
// issues are picked up after each round of merges.
//
// Usage:
//   npx tsx .sandcastle/main.mts
// Or add to package.json:
//   "scripts": { "sandcastle": "npx tsx .sandcastle/main.mts" }

import { execSync } from "node:child_process";

import * as sandcastle from "@ai-hero/sandcastle";
import { docker } from "@ai-hero/sandcastle/sandboxes/docker";
import { z } from "zod";

// The planner emits its plan as JSON inside <plan> tags; Output.object extracts
// and validates it against this schema. We use Zod here, but any Standard
// Schema validator works just as well — Valibot, ArkType, etc. See
// https://standardschema.dev.
const planSchema = z.object({
  issues: z.array(
    z.object({ id: z.string(), title: z.string(), branch: z.string() }),
  ),
});

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Maximum number of plan→execute→merge cycles before stopping.
// Raise this if your backlog is large; lower it for a quick smoke-test run.
const MAX_ITERATIONS = 10;

// Hooks run inside the sandbox before the agent starts each iteration.
// `uv sync` provisions the Python venv (and pinned toolchain) from the committed
// uv.lock, so the sandbox always has fresh, reproducible dependencies.
const hooks = {
  sandbox: { onSandboxReady: [{ command: "uv sync" }] },
};

// Nothing host-specific to copy in: the Python .venv is platform-specific
// (macOS host vs. Linux sandbox), so it is rebuilt by `uv sync` above rather
// than copied. uv's download cache keeps that fast.
const copyToWorktree: string[] = [];

// ---------------------------------------------------------------------------
// Merge-selection helpers
//
// A branch must be merged whenever it carries commits the base branch does not
// yet have — NOT only when the current iteration's implementer produced commits.
// Otherwise a branch whose work was fully committed in an earlier iteration (its
// implementer now correctly reports "no new commits") never lands: the issue
// stays open, the planner re-selects it next cycle, and the loop spins on it
// until MAX_ITERATIONS. These helpers let the merge phase pick up any open-issue
// branch that has unmerged commits, regardless of which iteration produced them.
// ---------------------------------------------------------------------------

// Run a host git/gh command and return its trimmed stdout.
function sh(cmd: string): string {
  return execSync(cmd, { encoding: "utf8" }).trim();
}

// The branch the loop merges into. Captured once: the merge phase merges into it
// and leaves it checked out, so it does not change across iterations.
const baseBranch = sh("git rev-parse --abbrev-ref HEAD");

// True if `branch` has commits the base branch does not yet contain.
function branchIsAhead(branch: string): boolean {
  try {
    return Number(sh(`git rev-list --count ${baseBranch}..${branch}`)) > 0;
  } catch {
    // Branch ref absent on the host (no commits ever synced) → nothing to merge.
    return false;
  }
}

// Local sandcastle/issue-<N> branches, each paired with its issue number.
function sandcastleIssueBranches(): { branch: string; id: string }[] {
  let out: string;
  try {
    out = sh("git for-each-ref --format='%(refname:short)' refs/heads/sandcastle/");
  } catch {
    return [];
  }
  const branches: { branch: string; id: string }[] = [];
  for (const branch of out.split("\n").filter(Boolean)) {
    const match = branch.match(/^sandcastle\/issue-(\d+)$/);
    if (match) branches.push({ branch, id: match[1]! });
  }
  return branches;
}

// Map of open-issue number → title, so the merge sweep can label stranded
// branches and never merges one whose issue is already closed. Empty on error,
// which safely degrades to merging only this iteration's planned branches.
function openIssueTitles(): Map<string, string> {
  const titles = new Map<string, string>();
  try {
    const json = sh("gh issue list --state open --limit 200 --json number,title");
    for (const issue of JSON.parse(json) as { number: number; title: string }[]) {
      titles.set(String(issue.number), issue.title);
    }
  } catch (err) {
    console.warn(`  Could not list open issues (${err}); merging planned branches only.`);
  }
  return titles;
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

for (let iteration = 1; iteration <= MAX_ITERATIONS; iteration++) {
  console.log(`\n=== Iteration ${iteration}/${MAX_ITERATIONS} ===\n`);

  // -------------------------------------------------------------------------
  // Phase 1: Plan
  //
  // The planning agent (opus, for deeper reasoning) reads the open issue list,
  // builds a dependency graph, and selects the issues that can be worked in
  // parallel right now (i.e., no blocking dependencies on other open issues).
  //
  // It outputs a <plan> JSON block — Output.object parses and validates it.
  // -------------------------------------------------------------------------
  const plan = await sandcastle.run({
    hooks,
    sandbox: docker(),
    name: "planner",
    // One iteration is enough: the planner just needs to read and reason,
    // not write code. (Structured output requires maxIterations: 1.)
    maxIterations: 1,
    // Opus for planning: dependency analysis benefits from deeper reasoning.
    agent: sandcastle.claudeCode("claude-opus-4-8"),
    promptFile: "./.sandcastle/plan-prompt.md",
    // Extract and validate the <plan> JSON into a typed object. Throws
    // StructuredOutputError if the tag is missing, the JSON is malformed, or
    // validation fails — which aborts the loop.
    output: sandcastle.Output.object({ tag: "plan", schema: planSchema }),
  });

  const issues = plan.output.issues;

  if (issues.length === 0) {
    // No unblocked work — either everything is done or everything is blocked.
    console.log("No unblocked issues to work on. Exiting.");
    break;
  }

  console.log(
    `Planning complete. ${issues.length} issue(s) to work in parallel:`,
  );
  for (const issue of issues) {
    console.log(`  ${issue.id}: ${issue.title} → ${issue.branch}`);
  }

  // -------------------------------------------------------------------------
  // Phase 2: Execute + Review
  //
  // For each issue, create a sandbox via createSandbox() so the implementer
  // and reviewer share the same sandbox instance per branch. The implementer
  // runs first; if it produces commits, the reviewer runs in the same sandbox.
  //
  // Promise.allSettled means one failing pipeline doesn't cancel the others.
  // -------------------------------------------------------------------------

  const settled = await Promise.allSettled(
    issues.map(async (issue) => {
      const sandbox = await sandcastle.createSandbox({
        branch: issue.branch,
        sandbox: docker(),
        hooks,
        copyToWorktree,
      });

      try {
        // Run the implementer
        const implement = await sandbox.run({
          name: "implementer",
          maxIterations: 100,
          agent: sandcastle.claudeCode("claude-opus-4-8"),
          promptFile: "./.sandcastle/implement-prompt.md",
          promptArgs: {
            TASK_ID: issue.id,
            ISSUE_TITLE: issue.title,
            BRANCH: issue.branch,
          },
        });

        // Only review if the implementer produced commits
        if (implement.commits.length > 0) {
          const review = await sandbox.run({
            name: "reviewer",
            maxIterations: 1,
            agent: sandcastle.claudeCode("claude-opus-4-7"),
            promptFile: "./.sandcastle/review-prompt.md",
            promptArgs: {
              BRANCH: issue.branch,
            },
          });

          // Merge commits from both runs so the merge phase sees all of them.
          // Each sandbox.run() only returns commits from its own run.
          return {
            ...review,
            commits: [...implement.commits, ...review.commits],
          };
        }

        return implement;
      } finally {
        await sandbox.close();
      }
    }),
  );

  // Log any agents that threw (network error, sandbox crash, etc.).
  for (const [i, outcome] of settled.entries()) {
    if (outcome.status === "rejected") {
      console.error(
        `  ✗ ${issues[i]!.id} (${issues[i]!.branch}) failed: ${outcome.reason}`,
      );
    }
  }

  // Decide what to merge. A branch belongs in the merge set when it carries
  // commits the base branch lacks — whether produced this iteration or
  // committed-but-never-merged in an earlier one. Keyed by issue id to dedupe.
  const mergeById = new Map<string, { id: string; title: string; branch: string }>();

  // This iteration's planned issues: include any whose pipeline fulfilled and
  // whose branch carries commits. The commits.length check preserves the
  // original signal; branchIsAhead additionally catches a branch fully committed
  // in a prior iteration whose implementer made no new commits this time.
  for (const [i, outcome] of settled.entries()) {
    if (outcome.status !== "fulfilled") continue;
    const issue = issues[i]!;
    if (outcome.value.commits.length > 0 || branchIsAhead(issue.branch)) {
      mergeById.set(issue.id, issue);
    }
  }

  // Sweep up any other open-issue branch with unmerged commits, so a branch
  // stranded by an earlier iteration lands even if the planner did not
  // re-select its issue this round.
  const openTitles = openIssueTitles();
  for (const { branch, id } of sandcastleIssueBranches()) {
    if (mergeById.has(id)) continue;
    const title = openTitles.get(id);
    if (title === undefined) continue; // closed/merged issue — leave it alone
    if (!branchIsAhead(branch)) continue;
    mergeById.set(id, { id, title, branch });
  }

  const completedIssues = [...mergeById.values()];
  const completedBranches = completedIssues.map((i) => i.branch);

  console.log(
    `\nExecution complete. ${completedBranches.length} branch(es) with commits:`,
  );
  for (const branch of completedBranches) {
    console.log(`  ${branch}`);
  }

  if (completedBranches.length === 0) {
    // No branch carries unmerged commits this cycle — nothing to merge.
    console.log("No unmerged commits on any open-issue branch. Nothing to merge.");
    continue;
  }

  // -------------------------------------------------------------------------
  // Phase 3: Merge
  //
  // One agent merges all completed branches into the current branch,
  // resolving any conflicts and running tests to confirm everything works.
  //
  // The {{BRANCHES}} and {{ISSUES}} prompt arguments are lists that the agent
  // uses to know which branches to merge and which issues to close.
  // -------------------------------------------------------------------------
  await sandcastle.run({
    hooks,
    sandbox: docker(),
    name: "merger",
    maxIterations: 1,
    agent: sandcastle.claudeCode("claude-opus-4-8"),
    promptFile: "./.sandcastle/merge-prompt.md",
    promptArgs: {
      // A markdown list of branch names, one per line.
      BRANCHES: completedBranches.map((b) => `- ${b}`).join("\n"),
      // A markdown list of issue IDs and titles, one per line.
      ISSUES: completedIssues.map((i) => `- ${i.id}: ${i.title}`).join("\n"),
    },
  });

  console.log("\nBranches merged.");
}

console.log("\nAll done.");
