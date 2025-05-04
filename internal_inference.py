import os
import argparse
import joblib
import librosa
import numpy as np
import pandas as pd
import logging
import time
import noisereduce as nr
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence

logging.basicConfig(level=logging.INFO)

# //////////////////// Preprocessing ////////////////////


def load_audio(file_path, target_sr=16000):
    if file_path.lower().endswith(".mp3"):
        audio = AudioSegment.from_file(file_path, format="mp3").set_channels(1)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        sr = audio.frame_rate
        y = librosa.resample(samples, orig_sr=sr, target_sr=target_sr)
        return y, target_sr
    else:
        y, sr = librosa.load(file_path, sr=target_sr)
        return y, sr


def normalize_volume(samples):
    return librosa.util.normalize(samples)


def reduce_noise(samples, sr):
    return nr.reduce_noise(y=samples, sr=sr)


def bandpass_filter(samples, sr, low=80, high=8000):
    fft = librosa.stft(samples)
    freqs = librosa.fft_frequencies(sr=sr)
    mask = (freqs >= low) & (freqs <= high)
    fft[~mask, :] = 0
    return librosa.istft(fft)


def remove_silence_from_array(y, sr, silence_thresh=-35, min_silence_len=300):
    temp_path = f"temp_for_silence_{os.getpid()}.wav"
    sf.write(temp_path, y, sr)

    sound = AudioSegment.from_wav(temp_path)
    chunks = split_on_silence(
        sound,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=100,
    )

    os.remove(temp_path)

    if not chunks:
        logging.warning("No speech detected")
        return y

    combined = AudioSegment.empty()
    for chunk in chunks:
        combined += chunk

    samples = np.array(combined.get_array_of_samples()).astype(np.float32) / 32768.0
    return samples


def preprocess_audio(file_path, output_file_path):
    logging.info(f"Processing: {file_path}")
    try:
        print(f"Current working directory: {os.getcwd()}")

        y, sr = load_audio(file_path)
        y = reduce_noise(y, sr)
        y = normalize_volume(y)
        y = bandpass_filter(y, sr)
        y = remove_silence_from_array(y, sr)
        sf.write(output_file_path, y, sr)
        logging.info(f"Saved preprocessed audio to: {output_file_path}")
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        raise


# //////////////////// Feature Extraction ////////////////////
class FeatureExtractor:
    def __init__(self, sr=22050, frame_size=2048, hop_length=512):
        self.sr = sr
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract_features(self, audio):
        features = {
            "pitch": np.mean(self.extract_pitch(audio)),
            "chroma": self.extract_chroma(audio),
            "mfcc": self.extract_mfcc(audio),
            "zcr": np.mean(self.extract_zcr(audio)),
        }
        return features

    def extract_zcr(self, audio):
        zcr = librosa.feature.zero_crossing_rate(
            audio, frame_length=self.frame_size, hop_length=self.hop_length
        )[0]
        return zcr if zcr.size > 0 else np.array([0.0])

    def extract_mfcc(self, audio, n_mfcc=13):
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_fft=self.frame_size,
            hop_length=self.hop_length,
            n_mfcc=n_mfcc,
        )
        return np.mean(mfccs, axis=1) if mfccs.size > 0 else np.zeros(n_mfcc)

    def extract_chroma(self, audio):
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=self.sr, n_fft=self.frame_size, hop_length=self.hop_length
        )
        return np.mean(chroma, axis=1) if chroma.size > 0 else np.zeros(12)

    def extract_pitch(self, audio):
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sr)
        pitch_values = pitches[magnitudes > 0]
        return pitch_values if pitch_values.size > 0 else np.array([0.0])


def extract_features_to_df(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None, mono=True, duration=5.0)

        if len(audio) < sr:
            logging.warning(
                f"Skipping {file_path}: Audio too short ({len(audio)/sr:.2f}s)"
            )
            return None

        extractor = FeatureExtractor(sr=sr)
        features = extractor.extract_features(audio)

        flat = {}
        for key, val in features.items():
            if isinstance(val, np.ndarray):
                for i, v in enumerate(val):
                    flat[f"{key}_{i}"] = v
            else:
                flat[key] = val

        return pd.DataFrame([flat])
    except Exception as e:
        logging.error(f"Failed to process {file_path}: {e}")
        return None


def main(input_path, output_path):
    try:
        model = joblib.load("./model/svm_model.pkl")
        scaler = joblib.load("./model/scaler.pkl")

        # Ensure output_path is a file (not just a directory)
        if os.path.isdir(output_path):
            base_name = os.path.basename(input_path)
            output_path = os.path.join(output_path, base_name)

        preprocess_audio(input_path, output_path)
        features_df = extract_features_to_df(output_path)

        if features_df is None:
            raise ValueError("Feature extraction failed.")

        scaled = scaler.transform(features_df)
        prediction = model.predict(scaled)
        print(prediction[0])

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Inference Script")
    parser.add_argument("--input", required=True, help="Path to the input audio file")
    parser.add_argument(
        "--output", required=True, help="Path to save the processed audio"
    )
    args = parser.parse_args()

    main(args.input, args.output)
