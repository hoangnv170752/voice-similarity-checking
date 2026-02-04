import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os

def extract_loudest_voice(file_path):
    """
    Extracts the loudest section of the audio (likely the primary speaker).
    """
    y, sr = librosa.load(file_path, sr=None, mono=True)

    frame_length = 2048
    hop_length = 512
    energy = np.array([
        sum(abs(y[i:i + frame_length]) ** 2)
        for i in range(0, len(y), hop_length)
    ])

    loudest_frame_idx = np.argmax(energy)
    start = loudest_frame_idx * hop_length
    end = start + frame_length

    y_loudest = y[start:end]
    return y_loudest, sr

def extract_features(audio_data, sr):
    """
    Extracts MFCC features from the audio data.
    """
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])

def compare_voices(file_paths):
    """
    Compares voices from multiple audio files and calculates similarity scores.
    """
    features = []
    file_names = [os.path.basename(path) for path in file_paths]

    for file_path in file_paths:
        print(f"Processing: {file_path}")
        loudest_audio, sr = extract_loudest_voice(file_path)
        features.append(extract_features(loudest_audio, sr))

    num_files = len(features)
    similarity_matrix = np.zeros((num_files, num_files))

    for i in range(num_files):
        for j in range(num_files):
            similarity_matrix[i, j] = cosine_similarity(
                [features[i]], [features[j]]
            )[0][0]

    return similarity_matrix, file_names

def plot_similarity_matrix(similarity_matrix, file_names, save_path='similarity_matrix.png'):
    """
    Visualizes the similarity matrix using a heatmap.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Similarity Score')
    plt.xticks(range(len(file_names)), file_names, rotation=45, ha='right')
    plt.yticks(range(len(file_names)), file_names)
    plt.title("Voice Similarity Matrix", fontsize=16)
    plt.xlabel("Audio Files", fontsize=12)
    plt.ylabel("Audio Files", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved similarity matrix to: {save_path}")
    plt.show()
    plt.close()

def plot_bar_chart_comparison(similarity_matrix, file_names, reference_idx=None, save_path='similarity_bar_chart.png'):
    """
    Creates a bar chart comparing all files against a reference file (default: original.wav or last file).
    """
    if reference_idx is None:
        if 'original.wav' in file_names:
            reference_idx = file_names.index('original.wav')
        else:
            reference_idx = len(file_names) - 1
    
    reference_name = file_names[reference_idx]
    similarities = []
    labels = []
    
    for i in range(len(file_names)):
        if i != reference_idx:
            similarities.append(similarity_matrix[reference_idx][i])
            labels.append(file_names[i])
    
    percentages = [round(sim * 100, 1) for sim in similarities]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(labels)), percentages, color='skyblue', edgecolor='black', alpha=0.8)
    
    for bar, percentage in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{percentage}%", ha='center', fontsize=11, color='black', fontweight='bold')
    
    plt.title(f"Voice Similarity as Percentages\n(Compared to {reference_name})", fontsize=16)
    plt.xlabel("Audio Files", fontsize=14)
    plt.ylabel("Similarity (%)", fontsize=14)
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved bar chart to: {save_path}")
    plt.show()
    plt.close()

def analyze_and_plot_voice_similarity(similarities, file_names, threshold=85):
    """
    Analyzes voice similarities, plots a bar chart, and groups files based on similarity.

    Parameters:
    - similarities: List of similarity scores (e.g., [0.85, 0.78, 0.92])
    - file_names: List of file names (e.g., ["Reference", "File 1", "File 2", "File 3"])
    - threshold: Percentage threshold for grouping voices as the same person
    """
    percentages = [round(sim * 100, 2) for sim in similarities]

    same_person_files = [file_names[0]]
    different_person_files = []

    for i, percentage in enumerate(percentages):
        if percentage >= threshold:
            same_person_files.append(file_names[i + 1])
        else:
            different_person_files.append(file_names[i + 1])

    print("\nVoice Similarity Analysis:")
    print(f"Threshold for Same Person: {threshold}%\n")
    print("Files Likely from the Same Person:")
    for file in same_person_files:
        print(f"- {file}")
    if different_person_files:
        print("\nFiles Likely from Different People:")
        for file in different_person_files:
            print(f"- {file}")

def main():
    """
    Main function to run voice similarity checking.
    """
    voices_folder = "voices"
    
    if not os.path.exists(voices_folder):
        print(f"Error: '{voices_folder}' folder not found.")
        print("Please create a 'voices' folder and add your audio files.")
        return
    
    audio_extensions = ('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac')
    file_paths = [
        os.path.join(voices_folder, f) 
        for f in os.listdir(voices_folder) 
        if f.lower().endswith(audio_extensions)
    ]
    
    if len(file_paths) == 0:
        print(f"No audio files found in '{voices_folder}' folder.")
        print(f"Supported formats: {', '.join(audio_extensions)}")
        return
    
    if len(file_paths) < 2:
        print("Need at least 2 audio files to compare.")
        return
    
    file_paths.sort()
    print(f"Found {len(file_paths)} audio files in '{voices_folder}' folder:")
    for fp in file_paths:
        print(f"  - {os.path.basename(fp)}")
    print()

    similarity_matrix, file_names = compare_voices(file_paths)
    
    print("\nSimilarity Matrix:")
    print(similarity_matrix)
    print()
    
    plot_bar_chart_comparison(similarity_matrix, file_names, save_path='similarity_bar_chart.png')
    
    plot_similarity_matrix(similarity_matrix, file_names, save_path='similarity_matrix.png')
    
    reference_idx = file_names.index('original.wav') if 'original.wav' in file_names else len(file_names) - 1
    similarities = [similarity_matrix[reference_idx][i] for i in range(len(file_names)) if i != reference_idx]
    reference_name = file_names[reference_idx]
    other_names = [file_names[i] for i in range(len(file_names)) if i != reference_idx]
    analyze_and_plot_voice_similarity(similarities, [reference_name] + other_names, threshold=85)
    
    print("\n✅ Completed! Generated 2 PNG files:")
    print("  1. similarity_bar_chart.png")
    print("  2. similarity_matrix.png")

if __name__ == "__main__":
    main()
