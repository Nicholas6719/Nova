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
| `tests/test_honesty.py` | no spoken promise or false "done"; typed turns stay silent |
| `tests/test_unprompted.py` | the single door, ceilings, restarts, standing watches |
| `tests/test_turnstats.py` | every turn recorded with its path; "why?" answered from the record |
| `tests/test_world_facts.py` | Stage 3: detection, the privacy guard (counted at the network), relevance, intake rules, the word guard, disputes, 429 handling, night school's audit / curiosity / gates / distillation, the real pipeline |
| `tests/world_facts_live.py` | Stage 3 with the real 3B, real Wikipedia and the real process (NOVA_TEST_LIVE=1) |

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
claims a fallen-through sentence, and **8a** (`_offer_brain_guess`) before the
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

**Stage 3 — answering from its own knowledge. DONE (2026-11-19).** He chose
**option C** of the three he was shown (A local only; B Wikipedia lookups
only when he asks; C B plus overnight research on his past questions). It is
invariant 3's fifth exception, written up in CLAUDE.md.

`world_facts.py` (store), `wiki.py` (the one source), `night_school.py`.
Pipeline: stage 2b‴ (receipts, disputes, the night report) and 8c (serve from
the store, otherwise look up and hand the answering sentences to the model).

What is different from JARVIS's fact store, each from a measurement:

- **Nova's LLM has no tools**, so nothing researches mid-answer. A general-
  knowledge question that reaches the LLM stage is looked up FIRST (one
  combined request: search, lead sections, page ids, revision ids) and the
  sentences that answer it — ranked locally by bge-small — go to the model as
  reference. Real 3B, nine canned questions: right 9/9 with the reference,
  8/9 without (it said the Eiffel Tower is "approximately 324 meters").
- **The word guard.** JARVIS serves at cosine 0.90. With Nova's embedder that
  would answer "who directed jaws 2" (0.937), "how wide is the mariana
  trench" (0.943) and "who starred in jaws" (0.921) with the wrong fact. A
  fact is served only if the question KIND (who / when / where / why / count
  / other) and the CONTENT WORDS also match, after a small synonym map. All
  ten measured near-misses are refused; three real paraphrases are missed,
  which costs a live lookup, never a wrong answer.
- **The 3B cannot judge "timeless".** JARVIS asks its 20B. Nova's 3B said YES
  to 6 of 8 changeable facts (Apple's CEO, the tallest building, Japan's
  population, the newest iPhone, the 100 m record, the last World Cup). The
  deterministic realm-2 screen (JARVIS's trigger words + present-tense "who
  is / holds / leads", "last", "won", growing counts) catches all 8; the
  model only ever vetoes.
- **Grounding.** Every number in a stored answer must be in the Wikipedia
  text, and 80% of its words must trace back to it — fact_reconciler's rule
  for what he says, applied to the world.
- **No "why" is stored.** Synthesised to one sentence, "why is the sky blue"
  became "due to a phenomenon called sky radiation": grounded, and no answer.
- **Relevance floor 0.70, not 0.80.** On real Wikipedia text the right pages
  score 0.795-0.898 and another question's pages at most 0.564; 0.80 would
  have discarded the Spielberg sentence for "who directed jaws".
- **Privacy guard before every request**: nothing first or second person, no
  name from his memory (relations, his own), no contact details. Tested by
  counting what reaches the network boundary.
- **The audit uses revision ids**: one request re-checks up to fifty facts,
  an unedited page is confirmed without reading it. An edited page that no
  longer contains the answer can be demoted by the model (CHANGED) but never
  confirmed — measured, the 3B said SAME for a fact against a sentence about
  cinema screens. Two strikes demote.
- **Silent**: night school never speaks. "What did you learn last night?",
  "how do you know that?" and "that's wrong" are how he sees and steers it.

Measured live (tests/world_facts_live.py): real lookups 0.84-0.91s; in the
real process a new question was looked up and answered (first word 2.7s, the
lookup ~1-2s of it before ranking was shortlisted to ≤14 sentences — ranking
alone fell from 220-640ms to 65-220ms with the top sentence unchanged on all
12 fixtures), stored ~3s later off the live path, and asked again another way
answered from the store in 22ms; the Apple CEO question was looked up live
(Wikipedia knew what the 3B could not) and never stored; the overnight audit
confirmed a stored fact against the live page.

Not verified: a real night (the window, idle and power gates are tested with
a clock; the run itself was forced in-process).

**Stage 4 — the rest of JARVIS. DONE (2026-11-19), local parts.**

Compared against JARVIS ARCHITECTURE.md "What it does without being asked" and
"Rules the code is built around"; what Nova lacked and now has:

- **Arithmetic and units in code** (`mathskill.py`, `units.py`, ported). Stage
  2e-0, before the brain. No `eval`: number words → digits, operator words →
  symbols, a small recursive-descent parser; a sentence is accepted only if
  EVERY word is part of the sum, so "set the volume to 50 percent" and "I have
  2 kids" never land there. Nova additions: "25% of 10%" answers "2.5 percent";
  tips on a dollar amount (15/18/20 percent, or the rate he names). Evidence:
  his own turns sent "what's 10 plus 10", "what's 25% of 10%", "what's the
  square root of 10" and a tip on a $95.82 dinner to the 3B.
- **Never a promise nothing keeps** (`honesty.py`, ported + extended). Nova's
  LLM runs no tools, so on that path EVERY "I'll keep an eye on it / remind
  you / let you know" is unbacked, and so is every "I've opened / set / sent"
  (the 3B once said "I've opened the coding projects folder for you"). Checked
  per sentence before it is spoken (streaming untouched); the first becomes the
  truth, later ones are dropped, and history stores what she SAID.
  `llm.honesty_filter: false` rolls back. Also fixed on the way: typed (silent)
  turns that reached the LLM were spoken aloud.
- **One door for anything unprompted** (`delivery.py`). Timers, the proactive
  monitor and watches all go through `deliver()`: a UNIQUE row is claimed and
  committed before a word is said (never twice — across threads and restarts);
  hard ceilings for unprompted kinds (`delivery.max_per_hour` 4,
  `max_per_day` 12, counted from the table); over a ceiling or in
  `quiet_hours` an item is SHOWN, never dropped; what he asked for (timers,
  watches) is never held by a ceiling. The proactive monitor's "never twice"
  was an in-memory set that a restart forgot — now it claims through the door.
- **Standing watches** (`standing.py`), stage 2e-0b. "Tell me when my battery
  is fully charged", "warn me if my CPU goes above 90", "what are you
  watching", "cancel all watches". Rows in `standing_watches`, checked every
  30s, one-shot, fired once through the door, survive restarts; a condition
  already true is answered now, not stored. **Local readings only** (battery,
  CPU, memory): a stock or weather watch would poll the network unprompted,
  which is his call.

Not brought across (specific to JARVIS's Windows PC, phone or scanner):
Telegram delivery, briefs, news significance, the scanner desk.

Tests: `test_unprompted.py` 36/36, `test_honesty.py` 31/31, corpus 289/289
(arithmetic and watches added regression-first).

**Stage 5 — from evidence. First round DONE (2026-11-19).**

The evidence was his 243 distinct real turns (Aug 2026) and what Nova replied
to each, replayed through `main` and through this branch
(`NOVA_DATA_DIR/brain_replay_report.md`, local only — it quotes him):

| his 229 labelled turns | correct | miss | wrong action |
| --- | --- | --- | --- |
| `main` | 201 | 23 | 5 |
| now | **220** | 7 | **2** |

The two wrong actions left are Nova hearing her own voice (one is caught today
by `_sounds_like_me`; the other was never her reply). What the replay showed
the model had said, and what now happens instead: "what's 10 times 10 minus
4" was answered with the clock; a tip on $95.82 was given as $14.29 (15 percent
is $14.37); "is music playing?" got "No, music is not playing" without
looking; "how long will it take for me to get to CVS?" got "I'm not aware of
your current location".

Fixed, each regression-first:
- **A statement is not a request** (`shape.py`, one definition shared by the
  brain's skills and the regex read stages): "I asked Nova what the weather
  was, it kept it on the screen..." fetched a forecast; "my battery died
  yesterday" read the battery; "Apple stock is way too expensive" fetched a
  quote. Definitions ("what does a CPU actually do", "what is a stock split")
  are not readings, and "tell me WHEN ..." is never a reading of now.
- A trailing courtesy defeated every launch/quit rule ("open spotify please"
  was refused as unsupported); "turn the sound down", "turn repeat on" and
  "hide everything on home" (claimed by MINIMIZE — every app) now land.
- **Skill-level, not just stage-level, correctness for tools.** Stage scoring
  hid two: "close, brave" reflexed as `tools.open_app` (right action, wrong
  story in the log and in "why?"), and — held-out — "play some jazz" as the
  fixed "play music" command (the WRONG action: resume whatever was on). A
  tool skill now refuses when the dry run says the action belongs to another
  skill; a new gate in `test_brain_intent` fails on any disagreement.

Beyond JARVIS:
- **Turn records** (`turnstats.py`, table `turn_stats`): every turn's path
  (reflex / regex stage / math / watch / phrase / ask / LLM ...), skill,
  confidence, matched example, time to decide, time to first word. Nova had no
  such record, which is why this round had to replay old text; the next one
  starts from what actually happened. Local, beside the conversation history.
- **"Why did you do that?"** — Nova explains her last decision from that
  record ("I took that as a request to tell you the time, because it was
  closest to something I already know: what time is it, 98 percent sure. It
  took me 5 milliseconds to decide."). Never invented.
- **"How often do you answer things by yourself?" / "what have you learned?"**
  — a report from the same record: share handled without the model, median
  time to first word for each, and the latest phrasing learned.

Ideas that need his OK (they change what Nova does unprompted or touch
privacy), not built:
- stock / weather standing watches (poll the network unprompted);
- a weekly spoken "here's what I learned" — unprompted speech;
- turning the `[brain-disagree]` log into a weekly review of regex stages the
  brain has fully matched, so they can be retired.

## Known gaps

Remaining misses both routers share: "do I have plans tonight", "what's the
stock price of Netflix", "give me a quote on Microsoft", "where did I save my
tax return", "go to reddit dot com". Real turns still missed: "Pull up
Spider-Man No Way Home in Disney Plus", "find me, tickets to Spider-Man brand
new day", "Open AMC in the browser" (none of these sites are in
browser_control.SITES), "You didn't close the brave browser, please close the
brave browser", "Nova, got to sleep" (a mishear of the sign-off).

Stage 3: explanations are looked up but never stored, and Wikipedia's lead
sections do not contain every answer ("why is the sky blue" gets colour-swatch
sentences; the model then answers from memory as before). Present-tense
"who's the author of X" is refused by the store on purpose (same shape as
"who's the director of the FBI") and gets a live lookup instead.

"open notes" goes to file search, not the Notes app — in both routers, from
file_intents treating "notes" as a file noun.

The pre-existing `echo` suite fails on `main` as well: the canceller removes
0.0 dB where it measured 27-36 dB when built — an environment regression,
unrelated to the brain, worth its own investigation.
