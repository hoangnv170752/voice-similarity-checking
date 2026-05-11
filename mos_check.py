import os
import torch
import torchaudio
import soundfile as sf
import numpy as np
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE

VOICES_DIR = "voices"
ORIGINAL = os.path.join(VOICES_DIR, "original.wav")

def load_wav(path, target_sr=16000):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)  # [channels, samples]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav

print("Loading SQUIM models...")
obj_model = SQUIM_OBJECTIVE.get_model()
subj_model = SQUIM_SUBJECTIVE.get_model()

wav_orig = load_wav(ORIGINAL)

others = sorted([
    os.path.join(VOICES_DIR, f)
    for f in os.listdir(VOICES_DIR)
    if f.lower().endswith(".wav") and f != "original.wav"
])

print(f"\nReference: {ORIGINAL}")
print(f"\n{'File':<35} {'MOS':>6}  {'PESQ':>6}  {'STOI':>6}")
print("-" * 60)

results = []
for path in others:
    wav = load_wav(path)
    with torch.no_grad():
        stoi, pesq, sisdr = obj_model(wav)
        mos = subj_model(wav, wav_orig)
    results.append({
        "name": os.path.basename(path),
        "mos": mos.item(),
        "pesq": pesq.item(),
        "stoi": stoi.item(),
    })

results.sort(key=lambda x: x["mos"], reverse=True)

for r in results:
    print(f"{r['name']:<35} {r['mos']:>6.3f}  {r['pesq']:>6.3f}  {r['stoi']:>6.3f}")

print(f"\n{'MOS'}: predicted Mean Opinion Score (1–5, higher = better quality)")
print(f"{'PESQ'}: Perceptual Evaluation of Speech Quality")
print(f"{'STOI'}: Short-Time Objective Intelligibility (0–1)")
