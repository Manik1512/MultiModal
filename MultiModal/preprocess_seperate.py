PATH_TO_DIRECTORY = '/home/manik/Downloads/deepfake_dataset/test/fake'
PATH_TO_OP_DIRECTORY = '/home/manik/Downloads/deepfake_dataset_preprocessed/test/fake'

import cv2
import dlib
import numpy as np
import os
from tqdm import tqdm

# Paths
face_predictor_path = "/home/manik/Documents/MultiModal/MultiModal/misc/shape_predictor_68_face_landmarks.dat"
mean_face_path = "/home/manik/Documents/MultiModal/MultiModal/misc/20words_mean_face.npy"

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_predictor_path)

# Load mean face landmarks if you plan to align later (not mandatory for mouth crop)
mean_face = np.load(mean_face_path)

def get_mouth_crop(frame, landmarks, margin=20):
    """Extract mouth region based on 68 landmarks"""
    mouth_points = landmarks[48:68]  # mouth region
    x_min = np.min(mouth_points[:, 0]) - margin
    x_max = np.max(mouth_points[:, 0]) + margin
    y_min = np.min(mouth_points[:, 1]) - margin
    y_max = np.max(mouth_points[:, 1]) + margin

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame.shape[1], x_max)
    y_max = min(frame.shape[0], y_max)

    return frame[y_min:y_max, x_min:x_max]

def process_video(input_path, output_path, margin=20):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video {input_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    pbar = tqdm(total=total_frames, desc=os.path.basename(input_path))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)

        if len(faces) > 0:
            face = max(faces, key=lambda r: r.width() * r.height())
            shape = predictor(gray, face)
            landmarks = np.array([(p.x, p.y) for p in shape.parts()])

            mouth_crop = get_mouth_crop(frame, landmarks, margin)
            if mouth_crop.size == 0:
                pbar.update(1)
                continue

            if writer is None:
                h, w = mouth_crop.shape[:2]
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            writer.write(mouth_crop)

        pbar.update(1)

    cap.release()
    if writer:
        writer.release()
    pbar.close()


def process_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    videos = [f for f in os.listdir(input_dir) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

    for video_name in videos:
        input_path = os.path.join(input_dir, video_name)
        output_path = os.path.join(output_dir, video_name)
        process_video(input_path, output_path)

    print("✅ All videos processed successfully!")


if __name__ == "__main__":
    input_dir = PATH_TO_DIRECTORY
    output_dir = PATH_TO_OP_DIRECTORY
    process_dataset(input_dir, output_dir)
