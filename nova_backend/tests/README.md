# Nova tests

One command:

```bash
python nova_backend/tests/run_tests.py --quick    # env + routing + loop (fast, silent)
python nova_backend/tests/run_tests.py --all      # everything (plays audio)
```

| Suite | What it proves | Fidelity |
|---|---|---|
| `verify_environment.py` | Every engine imports and MLX generates a token | real engines |
| `test_routing_corpus.py` | Every phrase that has broken Nova still routes correctly | real code, no side effects |
| `test_conversation_loop.py` | The conversation state machine: when Nova keeps listening vs returns to wake | real `_main_loop`, scripted mic |
| `smoke_launch.py` | The REAL process starts, answers a turn over HTTP, and shuts down cleanly | real process |
| `test_full_sweep.py` | Every subsystem against real system state (EventKit, SQLite, filesystem) | real code + real system |

The runner pins itself to the interpreter Nova actually uses (miniforge),
mirroring `locatePython()` in `BackendManager.swift`, so it does not matter
which `python` you type. Override with `NOVA_PYTHON=/path/to/python` if you ever
need to. It refuses to run under an interpreter without Nova's dependencies
rather than reporting green about an environment Nova never touches.

## Rules these encode

**1. The harness may not construct what the product constructs.**
Tests call `VoiceAssistant._init_state()` — the real initializer — never a
hand-copied field list. A harness that builds its own object tests its own
construction: that is how `_init_screen()` went missing from `__init__` for a
whole session while 130 checks stayed green.

**2. Side effects are stubbed; decisions are not.**
`tools.match()` performs as it matches, so probing routing once genuinely
launched Spotify. `NoSideEffects` replaces `subprocess.run` and `time.sleep`
only — every regex and alias lookup runs for real. Stubbing a whole handler
made the test *lie* (it reported a match where the real code returns `None`).

**3. Listener rules apply to every response, not per feature.**
`listener.py` checks every spoken string in the sweep for spoken paths,
markdown, third person, invented advice, invented numbers, and length. Most of
what has embarrassed us was audible instantly but scoped to one feature's test.

**4. Model-dependent checks are not trusted from a single run.**
A 3B is a sampling process. The calendar editorializing passed one run and came
back in real use. If a behaviour depends on the model, either sample it
repeatedly or — better — make the guarantee deterministic in code.

**5. A green run is not "Nova works".**
`run_tests.py` prints what it *could not* verify after every run. Anything
needing a real microphone, real speakers, or a TCC grant held by `NovaOS.app`
is outside what any of this can prove.

## Adding to the corpus

`adversarial_phrases.txt` is the regression net. When a phrase breaks Nova in
real use, add it there **before** fixing the bug, confirm the suite fails, then
fix. Format is `phrase  ==>  expected-stage`.
