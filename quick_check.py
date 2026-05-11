import os
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path

encoder = VoiceEncoder()

ORIGINAL = "voices/original.wav"
VOICES_DIR = "voices"

others = sorted([
    os.path.join(VOICES_DIR, f)
    for f in os.listdir(VOICES_DIR)
    if f.lower().endswith(".wav") and f != "original.wav"
])

print(f"Reference: {ORIGINAL}\n")

wav_orig = preprocess_wav(Path(ORIGINAL))
embed_orig = encoder.embed_utterance(wav_orig)

results = []
for path in others:
    wav = preprocess_wav(Path(path))
    embed = encoder.embed_utterance(wav)
    score = float(np.dot(embed_orig, embed) / (np.linalg.norm(embed_orig) * np.linalg.norm(embed)))
    results.append((os.path.basename(path), score))

results.sort(key=lambda x: x[1], reverse=True)

print(f"{'File':<35} {'Score':>8}  {'Verdict'}")
print("-" * 65)
for name, score in results:
    if score >= 0.90:
        verdict = "Very high – likely same speaker"
    elif score >= 0.80:
        verdict = "High – probably same speaker"
    elif score >= 0.70:
        verdict = "Moderate – uncertain"
    else:
        verdict = "Low – likely different speaker"
    print(f"{name:<35} {score * 100:>7.2f}%  {verdict}")
