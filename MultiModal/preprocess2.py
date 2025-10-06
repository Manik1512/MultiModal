import os
import cv2
import torch
import face_alignment
from tqdm import tqdm
import subprocess

# =========================
# Config
# =========================
dataset_root   = "/home/manik/Downloads/dataset2/FakeVideo-RealAudio/Caucasian (American)"
processed_root = "/home/manik/Downloads/dataset2_4_preprocessed/FakeVideo-RealAudio/Caucasian (American)"
mouth_size = 96
fps = 25.0

# =========================
# Init face alignment
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)

# =========================
# Extract mouth
# =========================
def extract_mouth(video_path, save_path, size=96, fps=25.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open {video_path}")
        return
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        preds = fa.get_landmarks(frame)
        if preds is not None:
            landmarks = preds[0]
            # Mouth landmarks 48–67
            x_min, y_min = landmarks[48:68, 0].min(), landmarks[48:68, 1].min()
            x_max, y_max = landmarks[48:68, 0].max(), landmarks[48:68, 1].max()

            w, h = x_max - x_min, y_max - y_min
            x_min, y_min = int(x_min - 0.3 * w), int(y_min - 0.3 * h)
            x_max, y_max = int(x_max + 0.3 * w), int(y_max + 0.3 * h)

            # Clamp
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(frame.shape[1], x_max), min(frame.shape[0], y_max)

            mouth = frame[y_min:y_max, x_min:x_max]
            if mouth.size == 0:
                continue

            mouth = cv2.resize(mouth, (size, size))
            frames.append(mouth)

    cap.release()

    if len(frames) == 0:
        print(f"⚠️ No mouth detected in {video_path}")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save processed video (no audio)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (size, size))
    for f in frames:
        out.write(f)
    out.release()

    # =========================
    # Save audio alongside processed video
    # =========================
    audio_path = os.path.splitext(save_path)[0] + ".wav"  # same name, .wav format
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"⚠️ Failed to extract audio from {video_path}: {e}")


# =========================
# Collect video paths
# =========================
video_exts = [".mp4", ".avi", ".mov", ".mkv"]
video_files = []

for root, dirs, files in os.walk(dataset_root):
    for file in files:
        if any(file.lower().endswith(ext) for ext in video_exts):
            input_path = os.path.join(root, file)
            save_path = input_path.replace(dataset_root, processed_root)
            video_files.append((input_path, save_path))

# =========================
# Process with progress bar
# =========================
for input_path, save_path in tqdm(video_files, desc="Processing videos"):
    if not os.path.exists(save_path):
        extract_mouth(input_path, save_path, size=mouth_size, fps=fps)
