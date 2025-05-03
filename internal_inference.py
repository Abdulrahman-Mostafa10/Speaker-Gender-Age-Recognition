import argparse
import joblib
import librosa
import numpy as np
import pandas as pd
import logging
import time


# //////////////////// Feature Extraction ////////////////////
class FeatureExtractor:
    def __init__(self, sr=22050, frame_size=2048, hop_length=512):
        self.sr = sr
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract_features(self, audio):
        features = {}

        features["pitch"] = np.mean(self.extract_pitch(audio))
        features["chroma"] = self.extract_chroma(audio)
        features["mfcc"] = self.extract_mfcc(audio)
        features["zcr"] = np.mean(self.extract_zcr(audio))

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
        start_time = time.time()
        audio, sr = librosa.load(file_path, sr=None, mono=True, duration=5.0)

        if len(audio) < sr:  # Skip if audio is shorter than 1 second
            logging.warning(
                f"Skipping {file_path}: Audio too short ({len(audio)/sr:.2f} seconds)"
            )
            return None

        extractor = FeatureExtractor(sr=sr)
        features = extractor.extract_features(audio)

        flattened_features = {}
        for key, val in features.items():
            if isinstance(val, np.ndarray) and val.size > 1:
                for i, v in enumerate(val):
                    flattened_features[f"{key}_{i}"] = v
            else:
                flattened_features[key] = val

        return pd.DataFrame([flattened_features])

    except Exception as e:
        logging.error(f"Failed to process {file_path}: {e}")
        return None


def main(input_file, output_dir):
    try:
        model = joblib.load("./model/svm_model.pkl")
        scaler = joblib.load("./model/scaler.pkl")

        # Extract features
        features_df = extract_features_to_df(input_file)

        if features_df is None:
            raise ValueError("Failed to extract features from the audio.")

        # Scale the features
        scaled_features = scaler.transform(features_df)

        # Make the prediction
        prediction = model.predict(scaled_features)

        print(prediction[0])  # Assuming model returns a single value

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Inference Script")
    parser.add_argument("--input", required=True, help="Path to the input audio file")
    parser.add_argument("--output", required=True, help="Path to the output directory")
    args = parser.parse_args()

    main(args.input, args.output)
