# Voice Similarity Check (MFCC + Cosine Similarity)

A lightweight notebook-based tool to compare **how similar two or more voice clips sound**.  
It extracts MFCC features from the **loudest segment** of each audio file and computes pairwise **cosine similarity**. A heatmap and simple grouping logic help visualize which clips are more alike.

> Built with `librosa`, `NumPy`, `scikit-learn`, and `matplotlib`.

---

## Features

- 🔊 **Auto-focus on speech**: finds the loudest segment in each file to reduce silence/noise effects  
- 🎛️ **Compact features**: 13-MFCCs (mean + std) → 26-dimensional vector per clip  
- 📈 **Similarity matrix**: cosine similarity across all files, visualized as a heatmap  
- 🧭 **Quick decision aid**: threshold-based grouping (e.g., ≥ 85% → likely same speaker)  
- 🧪 **Notebook-first workflow**: easy to tweak and extend

---

## Files in this repo

- `Voice_similarity_check.ipynb` — main notebook (feature extraction, similarity, plots)  
- Example audio files (demo voices collected from free online resources):  
  - `are-you-ready-90703.mp3`  
  - `medieval-gamer-voice-darkness-hunts-us-what-youve-learned-stay-226596.mp3`  
  - `medieval-gamer-voice-the-day-is-ours-226579.mp3`  

⚠️ **Note**: These audio files are **not personal recordings**. They are public demo voices collected from free websites and included only for **illustration**.

---

## Installation

Tested with Python 3.9–3.11.

```bash
# (recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install dependencies
pip install numpy librosa scikit-learn matplotlib soundfile
```

---

## Quick Start (Notebook)

1. Open `Voice_similarity_check.ipynb` in Jupyter or Google Colab.  
2. Update the `file_paths` list to point to your audio files, e.g.:

```python
file_paths = [
    "samples/are-you-ready-90703.mp3",
    "samples/medieval-gamer-voice-darkness-hunts-us-what-youve-learned-stay-226596.mp3",
    "samples/medieval-gamer-voice-the-day-is-ours-226579.mp3",
]
```

3. Run the notebook to:
   - Extract the loudest segment from each file  
   - Compute MFCC features  
   - Build and plot the similarity matrix  
   - (Optional) Run the threshold-based grouping analysis  

---

## How it works

1. **Loudest segment selection**  
   - Splits the waveform into frames  
   - Picks the frame with maximum energy (reducing silence effects)  

2. **Feature extraction**  
   - Computes **13 MFCCs** with `librosa`  
   - Aggregates via **mean** and **std** → 26-D feature vector  

3. **Similarity**  
   - Uses **cosine similarity** between feature vectors  
   - Produces an `N×N` similarity matrix + heatmap  

4. **Grouping (optional)**  
   - Converts similarity to percentages (×100)  
   - Flags files ≥ threshold (default 85%) as “likely same speaker”  

---

## Example Usage

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Extract features (functions in the notebook)
features = []
for path in file_paths:
    loudest_audio, sr = extract_loudest_voice(path)
    features.append(extract_features(loudest_audio, sr))

features = np.array(features)
sim_matrix = cosine_similarity(features, features)

plot_similarity_matrix(sim_matrix, [p.split("/")[-1] for p in file_paths])
```

---

## Interpreting results

- **Heatmap**: brighter (toward red) = more similar  
- **Cosine values**:  
  - ~**0.85–1.00**: strong similarity (possibly same speaker)  
  - <0.85: less similar, may be different speakers  

⚠️ Similarity depends on recording quality, mic, background noise, etc.

---

## Ethical Use

- Audio files included here are **public demo voices** from free websites.  
- This project is for **educational and research purposes only**.  
- Do **not** use it for impersonation, surveillance, or biometric identification.  
- Always obtain consent before analyzing real human recordings.  

---

## Roadmap

- [ ] Replace loudest-frame heuristic with **VAD-based segmentation**  
- [ ] Support averaging across multiple speech segments  
- [ ] Add a **CLI script** for batch comparisons  
- [ ] Experiment with **speaker embeddings** (x-vectors, ECAPA-TDNN)  

---

## Citation / Acknowledgment

If you use this repo in academic work, please cite:

> Mamun, A. A. (2025). *Voice Similarity Check: MFCC + Cosine Similarity (notebook).* GitHub Repository.

---

## License

MIT — see `LICENSE`.

---

## Author

**Abdullah Al Mamun**  
Machine Learning & Data Science Researcher  
GitHub: [@aamamun1](https://github.com/aamamun1)  
LinkedIn: [@aamamun234](https://www.linkedin.com/in/aamamun234/)  

---

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)  
![Librosa](https://img.shields.io/badge/Librosa-Audio-green)  
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)  
![Status](https://img.shields.io/badge/Status-Research%20Project-lightgrey)
