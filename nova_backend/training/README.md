# Training the "Nova" wake word (OpenWakeWord)

Goal: produce **`nova.onnx`**, a neural wake-word model for the phrase **"Nova"**,
and drop it into `nova_backend/`. Runtime inference is 100% local (onnxruntime);
this training step is a one-time model-generation job run on a free Colab GPU.

The Nova backend is already wired for it — `stt_engine` loads
`wake_word.oww_model_path` and scores it per frame. Today `wake_word.engine` is
`"transcript"` (the old Whisper scan); once `nova.onnx` exists and tests well,
flip it to `"openwakeword"`.

---

## Why Colab (and why it's still fully local afterward)

OpenWakeWord trains a custom model from **synthetic speech**: it uses Piper TTS
to generate thousands of "Nova" clips across many voices/pitches/speeds, mixes
in background noise + room reverb, and trains a small classifier. That needs
PyTorch + Piper + several GB of noise/negative datasets — heavy to set up
locally and slow without a GPU. Colab does it in ~1 hour on a free GPU.

The **output** is a small `.onnx` file that runs entirely on-device. Training in
Colab is offline model generation, not a runtime cloud dependency — the same
way Nova's Whisper/MLX models were originally downloaded once.

---

## Steps (do this in your browser)

1. **Open the official notebook in Colab:**
   https://colab.research.google.com/github/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb

2. **Enable the GPU:** Runtime → Change runtime type → Hardware accelerator → **GPU** → Save.

3. **Run the setup cells** at the top (they `pip install` openWakeWord + Piper and
   download the background/noise/negative datasets). Just run them in order.

4. **Set the target phrase + model name.** In the notebook's config cell, set:
   - `target_phrase` → **`["nova"]`**
   - `model_name` → **`nova`**

   Leave the dataset paths, augmentation, and training defaults as the notebook
   sets them — those point at the datasets it just downloaded. See
   [`nova_wake.yaml`](nova_wake.yaml) in this folder for the full parameter set
   with recommended values and comments if you want to tune.

5. **Run the generate → augment → train cells** in order. This is the ~1-hour part.
   It generates positive clips, augments them, then trains.

6. **Download the model.** When training finishes it writes `nova.onnx` (and a
   `.tflite`). Download the **`.onnx`**.

7. **Install it into Nova:**
   - Put the file at `nova_backend/nova.onnx` (that matches `oww_model_path`).
   - In `nova_backend/config.json`, set `"wake_word": { "engine": "openwakeword" }`.
   - Relaunch the app (no Xcode rebuild — Nova runs the backend from the repo).

8. **Test + tune.** Say "Nova" in a quiet room, then with a fan running. Tune in
   `config.json`:
   - Too many false triggers → raise `oww_threshold` (try 0.6–0.7) and/or
     `oww_trigger_level` (2 → 3).
   - Misses your real "Nova" → lower `oww_threshold` (try 0.4).
   - `oww_vad_threshold` (0.5) makes it ignore non-speech noise; lower to 0.3 if
     it feels sluggish to trigger, 0 to disable the speech gate entirely.

Rollback anytime: set `engine` back to `"transcript"`.

---

## Heads-up: "Nova" is a single word

Single-word wake phrases are the hardest case for a wake model — short, and easy
to confuse with everyday speech, so expect more false positives than a two-word
phrase. Mitigations, in order of effort:

1. Raise `oww_threshold` / `oww_trigger_level` (config only, no retrain).
2. Add more `custom_negative_phrases` and retrain (words that sound like "Nova":
   "over", "nova scotia", "no", "innovate", "nova the name", etc.).
3. Retrain with `target_phrase: ["nova", "hey nova"]` so "hey nova" also works
   and gives the model more acoustic context — your call; you chose single-word
   "Nova", this is just the escape hatch if it's too trigger-happy.

---

## Local alternative (not recommended, but self-contained)

If you ever want to avoid Colab entirely, the same job runs via the installed
CLI using [`nova_wake.yaml`](nova_wake.yaml):

```bash
# one-time heavy setup: torch, piper-sample-generator, and the negative/noise
# datasets the yaml points at (several GB) — see the notebook's setup cells for
# the exact dataset URLs and expected paths.
python -m openwakeword.train --training_config nova_wake.yaml --generate_clips --augment_clips --train_model
```

Expect this to be much slower than Colab on a Mac (no CUDA), which is why Colab
is the recommended path.
