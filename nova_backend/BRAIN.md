# Nova's Brain

A learned intent router between speech-to-text and the LLM, ported from JARVIS
(MiniPCJarvis `docs/BRAIN.md`, `docs/BRAIN_ROADMAP.md`). This file covers the
port: what was brought across, what is different on Nova and why, the
numbers, and where each stage of the roadmap stands. For the original design
and JARVIS's history, read JARVIS's BRAIN.md; Nova's CLAUDE.md and invariants
win over anything there.

## Contents

1. [Where it sits](#where-it-sits)
2. [How a sentence is decided](#how-a-sentence-is-decided)
3. [Skills](#skills)
4. [What is different from JARVIS, and why](#what-is-different-from-jarvis-and-why)
5. [Tests](#tests)
6. [Numbers](#numbers)
7. [Roadmap status](#roadmap-status)
8. [Known gaps](#known-gaps)

---

## Where it sits

`nova.py`, `_handle_turn_impl`, stage **2e**: after system commands, the power
confirmation, the pending calendar/file/actuation questions and the anchored
view regex (2c), and in front of every regex command stage, which now live in
`_regex_commands` (2f-8) unchanged and in their old order.

```
2e  brain.decide(text)
      confident skill, slots extract  -> skill.run(...)        a REFLEX, no LLM
      confidently `general`           -> skip 2f-8, go to 8b   (conversation)
      anything else                   -> 2f-8 as before        (fallthrough)
```

A reflex runs the same `_run_*` stage method (`_run_calendar`, `_run_files`,
`_finish_tool`, ...) the regex stage would have run after its own match, so
what happens AFTER a decision — panels, follow-up offers, work mode, the
status row — cannot differ by who decided.

Rollback: `brain.enabled: false` in config.json. The regex chain then runs
exactly as it did before the brain existed. Until the brain has finished
loading (first run: ~7s to embed the seeds; later runs read them from
`nova_memory.db`), every turn takes the regex chain.

## How a sentence is decided

1. **Normalise** (`brain._light`, `_norm`): lowercase; strip a leading "Nova"
   (only when something follows it, so "hey Nova" stays a greeting) and
   politeness ("please", "can you", "for me"); "you tube" → "youtube"; digits
   and times → `N`. Deliberately minimal — BRAIN.md: add a canon rule only when
   a real sentence needs it. None of JARVIS's canon rules were brought across;
   each encodes a JARVIS feature.
2. **Embed** with `BAAI/bge-small-en-v1.5` through fastembed (local ONNX,
   single-threaded, ~4ms). The model (~64MB) downloads once into
   `nova_backend/models/` (gitignored); after that the brain never touches the
   network (`local_files_only` is tried first).
3. **Classify** — cosine against every example. The rule is JARVIS's, copied
   exactly:
   ```
   margin = top - rival          # rival = best example of ANOTHER skill
   confidence = top if margin >= 0.06 else top - (0.06 - margin) * 3.0
   ```
4. **Decide** — at or above `threshold` (0.82) the skill's slots run. If they
   return arguments, it is a reflex. If they refuse, the runner-up skill gets a
   turn (at most three). Below threshold, a match with raw top ≥ 0.66 whose
   slots extract is kept in `brain.unsure` — Stage 2's "did you mean ...?".

Thresholds live in config.json under `brain`. They are JARVIS's values, tuned
on real speech with this same embedder; that is why Nova uses bge-small and not
the MiniLM model Chroma already ships. Change them only with measurements.

### Confidence gate

`confidence.py` asks "was the transcript right?", the brain asks "which intent
is this?" — independent questions. `_confidence_gate` passes the matched
skill's tier as `min_tier`; the bar is the stricter of the two. The brain can
raise it (a file move the regex tier read as chat), never lower it.

## Skills

`skills.py`: 92 skills, ~680 seeds, most taken from the routing corpus. Each:

| Field | Meaning |
| --- | --- |
| `seeds` | example phrasings |
| `slots(va, text)` | arguments, or None to refuse; reuses the *_intents extractors |
| `run(va, text, args)` | the shared stage method |
| `tier` | confidence.py consequence: low / medium / high |
| `gate(va)` | the SITUATION (work mode), kept apart from the sentence |
| `label` | what "did you mean ...?" will say (Stage 2) |

**The slot rule is asymmetric, on purpose.** The brain may widen how Nova
understands a *question*; in Stage 1 it does not widen what makes her *change*
something unconfirmed:

- reads (calendar, weather, screen, market, folder listings, system readings,
  what's playing) are lenient — the embedding decides;
- unconfirmed writes (calendar writes, power, lock, launching or quitting an
  app by name, web searches, screenshots) are strict — the module's own regex
  must accept the sentence (`tools.detect()` for tools);
- file move / copy / rename are lenient because file_intents asks "move it to
  Documents?" and nothing on disk changes without a spoken yes;
- actuation is strict and work-mode gated, exactly as before.

**Shape guards** — each added after a measured wrong action, all the same
lesson CLAUDE.md records for the regex tools ("command words that are also
ordinary English must be shaped like commands"):

- a read must be shaped like a request: "my next meeting is at three" sat 0.89
  from "when's my next meeting" and read his calendar;
- navigation must be shaped like navigation: "we should go home soon" went
  home, "what's on the menu for dinner" opened the menu, "minimize Xcode"
  entered work mode;
- a state-changing tool must name its object or be claimed by tools itself:
  "what's next" skipped his song.

**Argument-free tools** carry a canonical command in tools' own vocabulary
("pause the music") and run `tools.match(command)` — exactly what the regex
path runs for that phrase. tools.match() has no intent names, so this is how a
skill reuses it without a second copy of its logic.

**`general`** has no runner. It is seeded with every `llm`, `unsupported` and
`no-signoff` line of the corpus plus conversation that uses command nouns as
topics ("my memory is terrible", "the menu at that restaurant was huge"). When
it wins confidently, Nova treats the turn as conversation.

## What is different from JARVIS, and why

| JARVIS | Nova | Why |
| --- | --- | --- |
| async (`asyncio.to_thread`) | synchronous, one RLock | Nova's pipeline is threaded |
| first-match canon rewrites | one placeholder rule, all applied | no Nova sentence has needed more |
| context nudges (stage, project, office) | none | no Nova evidence yet |
| `general_level` blocks web search | confident `general` skips regex command stages | Nova's failure was regex stealing conversation, not tool overuse |
| slots take light text | slots take the ORIGINAL text | Nova's extractors read capitals (places, companies) and digits |
| one long-lived connection | short-lived connection per write | JARVIS mistake #1, structurally impossible here |
| situational filters inside slots | separate `gate` | a load-time executability check outside work mode would otherwise delete learned actuation rows |

Kept exactly: the confidence/margin rule and thresholds, guards handing over to
the runner-up, the near miss kept rather than thrown away, seeds as ground
truth at load, placeholder stand-ins in `_executable` (mistake #2), copied
matrices (mistake #9).

## Tests

| Suite | Proves |
| --- | --- |
| `tests/test_brain_seeds.py` | no two skills own a canonical seed; every seed executable by its own skill; held-out overlap reported |
| `tests/test_brain_intent.py` | brain vs regex vs hybrid on: corpus, 5-fold held-out corpus, paraphrases v1, sealed paraphrases v2, his real turns (local only). Gates: hybrid ≥ regex on the corpus; no NEW wrong action on any set |
| `tests/test_brain_learning.py` | Stage 2 through the real pipeline: the learning bar, persistence (placeholder rows survive), no lingering SQLite lock, regex-taught reflexes, did-you-mean yes/no/restart, taught phrases, corrections |
| `tests/brain_live_check.py` | inside the real process: load, reflex timings, LLM still answers after reflexes, a did-you-mean round trip, a taught phrase, a correction |
| `tests/test_routing_corpus.py` | the regex chain alone (the `regex` column) |

Scoring counts a WRONG action (did something, not what he meant) apart from a
miss (fell to conversation or an honest refusal). Sentences whose canonical form
equals a seed are excluded from held-out scores and counted.

His real turns live in `NOVA_DATA_DIR/brain_real_utterances.txt`, hand
labelled, never committed (the repo is public). `?` marks a mishear with no
right answer.

## Numbers

Measured 2026-11-19. "main" is the chain before this work, scored with the
faithful harness (its own harness said 257/257).

| Set | main regex | branch regex | brain alone | hybrid (ships) |
| --- | --- | --- | --- | --- |
| corpus (272) | 254 | 272 | 264 | **272** |
| corpus, 5-fold held-out | — | 272 | 198 | **272**, 0 wrong |
| paraphrases v1 (145 not seeds) | — | 110 (75.9%) | 99 | **124 (85.5%)** |
| paraphrases v2, sealed (93) | — | 62 (66.7%) | 67 | **79 (84.9%)** |
| his real turns (108 not seeds) | — | 99 (91.7%) | 86 | **100 (92.6%)** |

No set shows a wrong action the regex chain did not already make. The 5 wrong
actions left on v2 are all the regex's own (see Known gaps).

Live process (`brain_live_check.py`): load + first embed 7s, reflex decisions
5-45ms (the high end is tools.detect inside a guard), answers through the real
handlers; LLM turns after reflexes answered normally.

"Brain alone" trails regex because 0.82 is strict and the seeds are thin; the
hybrid is what ships, and Stage 2's learning is how "brain alone" grows.

## Roadmap status

**Stage 1 — the router. DONE (2026-11-19).** All of the above.

**Stage 2 — learning and honest uncertainty. DONE (2026-11-19).**

Where it runs: stage **2b′** (`_brain_lessons`, before navigation, so none of
these can be re-read as a new command), the learning hook after a regex stage
claims a fallen-through sentence, and **8c** (`_offer_brain_guess`) before the
unsupported-action refusal.

- **Learning** (`Brain.learn`). Nova's LLM has NO tools, so JARVIS's teacher
  ("the LLM solved it with exactly one tool") does not exist here. The teacher
  is the regex chain behind the brain: a sentence the brain fell through on
  (not "surely conversation") that a regex stage then claimed.
  `skills.skill_for_regex` maps the claim to ONE skill (tools via the dry-run
  label's family, intersected with the brain's own best tools guess). The bar,
  each from a JARVIS mistake: ≤ 14 words; never a bare answer; the skill's own
  slots must execute BOTH the sentence and the canonical form that will be
  stored (mistake #2); never a seed's canonical form; not when `general` was
  ≥ 0.7; and — Nova's addition — **two judges must agree**: the brain's own
  rank-1 guess must already be that skill, because the regex makes its own
  wrong calls ("my battery died on the way to work" claims a battery reading).
- **"Did you mean <label>?"** when nothing claimed the sentence but the brain
  nearly knew: raw top ≥ `ask_threshold` 0.66 (JARVIS) AND margin-adjusted
  confidence ≥ `ask_min_confidence` 0.70 (Nova). With JARVIS's bar alone, all
  6 offers on his real turns were wrong; 0.70 separated useful from useless on
  everything measured (9 good kept, 12 bad dropped, 3 good lost — tuned on that
  data, re-measure as turns accumulate). Yes → runs the skill and stores the
  sentence as `source=user`. No → `reject()`, persisted, keyed on sentence AND
  skill. "No, I meant X" → reject, then route X. Anything else drops the
  question and routes normally.
- **Rejections now bind decide() too**, not only offers: a skill he said no to
  for a sentence is excluded from that sentence's candidates for good.
- **Taught phrases** (`brain_commands`): "when I say status check, what's my
  battery at and what time is it". Every step must be something Nova can
  already do (brain or a regex stage), nothing high-consequence (no shutdown
  behind a shortcut), at most six steps. Matched near-exact only (cosine ≥ 0.92
  or exact) because a phrase runs several actions; steps run through the brain,
  else the regex stages, never the LLM. "What phrases have I taught you",
  "forget the phrase status check". The confidence gate treats a phrase as
  consequential as its strictest step.
- **Corrections**: within 90s of a reflex, "no, I meant X" / "that's wrong"
  rejects that sentence for that skill and un-learns the example behind it if
  it was learned or taught (`unlearn`). Seeds are never deleted; the rejection
  keeps the sentence away from them. Skipped while an actuation question is
  open — "no, the other button" answers that.

Measured: `test_brain_learning.py` 44/44 (driven through the real
`_handle_turn_impl`); "did you mean" fires on 0 conversation sentences in the
corpus (78), his real turns (67), and both paraphrase sets (30, 20);
`brain_live_check.py` runs the full did-you-mean → yes → reflex round trip, a
taught phrase and a correction inside the real process.

**Stage 3 — answering from its own knowledge.** The fact store (timeless,
web-verified facts with sources) and night school (re-verify facts, research
questions answered from memory, widen routing with paraphrases). Paraphrase
distillation is local and allowed. Web verification and overnight research
need the network UNPROMPTED — beyond invariant 3's exceptions. That is his
decision; ask before building it.

**Stage 4 — the rest of JARVIS.** One door for anything unprompted (timers,
reminders, watches) with de-dup and hourly/daily ceilings; never speaking a
promise nothing will keep; standing "tell me when ..." watches that survive a
restart; arithmetic and unit conversion in code.

**Stage 5 — better than JARVIS.** From evidence: the `[brain-disagree]` log
and his real turns.

## Known gaps

Regex false positives the brain does not yet veto (its `general` match is
below 0.82 on them), each a wrong ACTION:
- "my battery died on the way to work" → battery reading
- "hide everything on the home screen" → tools minimises every app
- "what's the weather like on Mars" / "the weather has been so weird this
  year" → a forecast
- "I'm thinking of buying Tesla stock" / "what is a stock split" → a quote
- "what does a CPU actually do" → a CPU reading

Misses both routers share: "do I have plans tonight", "when is my next
appointment", "turn the sound up", "turn shuffle on", "open Xcode please"
(trailing "please" defeats the launch regex), "tell me what app I have open in
front" (claimed by tools' running-apps instead of screen).

"open notes" goes to file search, not the Notes app — in both routers, from
file_intents treating "notes" as a file noun.
