# Methodology: YOLO-G + AdvGRL Integration

A narrative of how we approached porting domain-adaptation improvements from "DA-Detect" (Paper 2) into the YOLO-G fork, separated from the code itself. Read this before re-reading any of the plans or specs — it explains the *why* behind the structure.

---

## The goal

YOLO-G already had a working image-level domain-adversarial branch (a DANN classifier wired into a YOLOv5-L backbone via a Gradient Reversal Layer). DA-Detect added several improvements on top: a dynamic adversarial weight (AdvGRL), an auxiliary "rainy" domain via RainMix augmentation, and an image-level triplet loss. We want all of those landed in YOLO-G — but as *independently togglable knobs*, not as one tangled patch.

The underlying reason is comparison. By the end of the work, we want to look at a table that says "feature X added Y mAP on top of the baseline" for each feature in isolation, then "feature X+Y combined added Z mAP." If we lump everything in at once, that table is impossible to produce; we'd be left with a single combined result and no way to attribute credit.

---

## Core principle: one feature, one flag, one PR

Each new capability is added as a separate command-line flag (`--da-img`, `--advgrl`, `--aux`, `--triplet-img`, `--da-img-warmup`). The flags default to off, so the training script's default behavior is unchanged. Each feature lands in its own pull request, in dependency order: things later flags need (e.g. the auxiliary domain loader) ship first.

This has three downstream effects:

1. **Reviewability.** Each PR touches a focused slice of the code. A reviewer can hold the change in their head.
2. **Bisectability.** If something regresses, we know which PR introduced it — there's only one component per PR.
3. **Ablation for free.** The final paper-style ablation table is just "run training with different combinations of flags." No code branches to maintain.

This is more work upfront than a single megapatch — you have to think about boundaries between features, design flags so they compose, and re-run sanity checks per PR — but it's cheaper than untangling a monolithic change after the fact.

---

## The "dumb model" refactor (PR 1)

Before any new feature work, the very first PR did something that looks like backwards motion: we *removed* domain-adaptation logic from the model and pushed it into the training script. The original YOLO-G model had the GRL and DA classifier baked into its forward pass. That made the model implicitly DA-aware and made any new DA component a model surgery.

The refactor decouples them:

- The model becomes "dumb." It outputs detection predictions plus an intermediate backbone feature. That's it.
- The training script owns the DA components. It constructs the DA classifier, applies the GRL with whatever weight strategy the flags ask for, computes the DA loss, and adds it to the detection loss before backprop.

After this refactor, adding a new DA feature is a training-script change, not a model change. The model stays stable across all subsequent PRs.

This was the most "infrastructure-heavy" PR and produces no visible improvement on its own. We accepted that because everything downstream is faster and safer once it lands.

---

## The sanity gate

Every PR is followed by a 10-epoch training run on Foggy Cityscapes, the standard adversarial-weather target. PR 1's source-only baseline came out at mAP@0.5 = 0.379. That number is the reference all later PRs are measured against.

The pass criterion is intentionally generous: "final mAP@0.5 within ±1.0 of 0.379." We don't want PRs to fail because of noise in a 10-epoch run, and we don't want to wait for full convergence on every gate. The point isn't to prove a feature is good; it's to detect catastrophic regressions and code-wiring bugs.

What counts as a failure: NaN losses, mAP collapsing to near-zero, the validator crashing. What does *not* count as a failure: mAP being slightly worse than baseline (DA features sometimes hurt source-domain performance; that's expected and is what the final paper-style evaluation will quantify properly).

We deliberately keep these runs short and cheap. The real evaluation (paper-faithful, full training) only happens once everything is wired up.

---

## Test-driven development, surgically applied

Every new helper — a function that computes a loss, a function that schedules a warmup weight, a function that pairs source and target batches — gets a unit test file *before* it gets wired into the training loop. The tests cover the boundaries (negative inputs, zero inputs, large inputs) and the happy path.

We don't write tests for the training loop itself; that's what the sanity gate is for. The split is deliberate: small pure functions are cheap to test in isolation and the tests pay for themselves the first time you refactor; integration logic is hard to test in isolation and is cheaper to verify end-to-end.

---

## Subagent-driven execution

For each PR, we write a plan that breaks the work into self-contained tasks (typically 3–6 tasks per PR). Each task is then executed by a fresh subagent — a new conversation with no memory of prior tasks — given just the task text and the surrounding context it needs.

After each task, two reviews run in sequence:

1. **Spec compliance review** — did the implementation match what the task asked for? Nothing added, nothing missing.
2. **Code quality review** — is the code clean, are there hidden bugs, are the tests meaningful?

If either review finds issues, the same subagent fixes them and the review repeats. Only when both reviews approve does the task close.

The motivation is twofold. First, fresh subagents don't carry over assumptions from earlier tasks; they only know what they're told, which forces task descriptions to be self-contained and complete. Second, splitting "do the work" from "check the work" means the implementer can't talk itself into shipping something half-done.

This is slower than just-implementing-it (every task has 2–3 review iterations on average) but the bugs that get caught are bugs that would otherwise show up in the sanity gate, where the feedback loop is six hours long instead of three minutes.

---

## When the sanity gate failed: PR 3.5

PR 2 (`--da-img`) and PR 3 (`--advgrl`) shipped clean code, passed all unit tests, and then catastrophically failed their sanity gates: mAP@0.5 collapsed from the 0.379 baseline to ~0.030 — over 90% loss.

The first instinct in this situation is to assume the code is wrong and to dig through the training step. We resisted that. Instead we asked: "Does the *original* YOLO-G (before our refactor) produce the same collapse, or did our PRs introduce the failure?" The answer turned out to be: the original YOLO-G never had a DA warmup, and applying full DA loss from iteration 0 destabilizes training while the detection head is still finding its footing. Our PR 2 was a faithful reproduction; the failure mode is inherent to the original design.

That reframing changed the response. Instead of debugging PR 2/3, we added a *new* PR — PR 3.5 — that introduces a DA-loss warmup as another togglable flag (`--da-img-warmup={off,gate,ramp}`). The default is `off`, which preserves the faithful original behavior; the new modes are opt-in. We didn't retroactively modify PR 2 or PR 3.

This matters for the final ablation. We need to be able to report both "DA-Detect's components without the warmup" (the faithful comparison against published numbers) and "DA-Detect's components with the warmup" (the practical, stable variant). If we had silently bundled the warmup into PR 2, that comparison would be impossible.

The PR 3.5 sanity gate ran three configurations. The `gate` mode (DA off for the warmup period, then a hard transition to on) collapsed the same way PR 2 did — the cliff at the warmup boundary is exactly the destabilizing event we were trying to avoid. The `ramp` mode (DA loss linearly increasing from 0 to 1 over the warmup window) appeared to work on the first run: mAP came in at 0.396, slightly *better* than the source-only baseline. The third configuration (ramp combined with AdvGRL) failed on its first run.

When we re-ran the supposedly-passing `ramp` configuration on a different cloud platform (Modal instead of Kaggle, same code, same hyperparameters, same data), it also collapsed. That broke the conclusion. The DA training regime at 10 epochs is on a knife edge — small platform-level differences (different GPU, different CUDA, different nondeterminism in BatchNorm) flip the same code between PASS and FAIL. The earlier 0.396 was a lucky outlier, not a robust fix. A targeted follow-up experiment (halving the GRL weight) showed the magnitude wasn't the driver — collapse was just delayed by one epoch, suggesting the real issue is *timing* (DA loss reaching full strength right at the LR-warmup peak, when the network is most fragile).

That finding changed how we think about the sanity gate. The gate was designed to catch "did this PR introduce a regression vs the source-only baseline." It still does that, but only at the resolution of "did training crash or go to NaN." A noisy ±1.0 mAP threshold over a single 10-epoch run cannot reliably distinguish DA variants in this regime — and chasing that signal would burn weeks of GPU time before we even have all the toggleable components in place.

So we changed the rule. Gates after PR 3.5 are smoke tests: does the code wire up, does training run for 10 epochs without crashing, do the loss curves look bounded. They do *not* attempt to rank variants by final mAP. The real comparison happens at the end, after all components are wired up, in the ablation phase — at full training length where the system has time to recover from warmup-peak transients, with multiple seeds where appropriate. The stabilization investigation (different ramp lengths, lower lambdas, backbone freezing) is deferred to that phase too, because it only matters if it changes the ablation story.

The takeaway isn't just "ramp beats gate" or "Kaggle beats Modal." It's that having every component as a toggle let us discover these things cheaply, that the methodology survived being wrong in public, and that we adjusted scope rather than burying the finding.

---

## What this all buys us

By the end of the work, every component — base DA, AdvGRL, RainMix auxiliary domain, image triplet loss, DA-loss warmup — is a flag on a training script. The same script with no flags reproduces the source-only baseline. With all flags on, it reproduces our best result. With any subset on, it produces an ablation point.

The ablation table at the end of the project isn't an extra deliverable. It falls out of the structure. That was the goal from the start.

---

## Things we are explicitly not doing

- **No "while we're in here" cleanup.** Each PR is scoped to one feature. Tempting refactors of nearby code wait for their own PR.
- **No silent default changes.** Every existing flag's default behavior is preserved across PRs. New behavior is opt-in.
- **No hiding failures.** When PR 2 collapsed we wrote it down, kept the run artifacts, and built PR 3.5 *on top of* the failure rather than papering over it.
- **No backwards-compat shims.** Once the dumb-model refactor landed, the old model interface was removed. We don't carry two code paths.
