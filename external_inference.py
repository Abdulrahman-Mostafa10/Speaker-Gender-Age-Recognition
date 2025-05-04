import os
import subprocess
import time

# Define the hardcoded directories
DATA_DIR = "./data"
PROCESSED_DIR = "./processed_data"
OUTPUT_DIR = "./output"
TIME_FILE = os.path.join(OUTPUT_DIR, "time.txt")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results.txt")
internal_inference_path = os.path.abspath("internal_inference.py")


# Ensure necessary directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_internal_inference(file_path, processed_path):
    start_time = time.time()

    try:
        result = subprocess.run(
            [
                "python",
                "internal_inference.py",
                "--input",
                file_path,
                "--output",
                processed_path,
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.dirname(
                internal_inference_path
            ),  # Raise exception if the script fails
        )

        prediction = result.stdout.strip()

    except subprocess.CalledProcessError as e:
        # If an error occurs in internal_inference.py
        prediction = f"Error with {file_path}: {e.stderr.strip()}"

    inference_time = time.time() - start_time

    # Write time
    with open(TIME_FILE, "a") as time_file:
        time_file.write(f"{inference_time:.3f}\n")

    # Write prediction
    with open(RESULTS_FILE, "a") as results_file:
        results_file.write(f"{os.path.basename(file_path)}: {prediction}\n")


def process_files_in_data_dir():
    for filename in os.listdir(DATA_DIR):
        if filename.lower().endswith((".wav", ".mp3")):
            file_path = os.path.join(DATA_DIR, filename)
            processed_path = os.path.join(
                PROCESSED_DIR, filename.replace(".mp3", ".wav")
            )

            run_internal_inference(file_path, processed_path)


if __name__ == "__main__":
    process_files_in_data_dir()
    print("Inference complete! Time and results have been logged.")
