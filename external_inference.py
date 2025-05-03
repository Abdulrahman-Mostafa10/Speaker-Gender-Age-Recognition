import os
import subprocess
import time

# Define the hardcoded directories
DATA_DIR = "./data"  # Path to the data directory
TIME_FILE = "./output/time.txt"  # File to store the time taken for each inference
RESULTS_FILE = "./output/results.txt"  # File to store the predictions


def run_internal_inference(file_path):
    # Call the internal inference script for each file
    start_time = time.time()

    # Run the internal inference script and pass the file path as an argument
    result = subprocess.run(
        [
            "python",
            "internal_inference.py",
            "--input",
            file_path,
            "--output",
            "output/",
        ],
        capture_output=True,
        text=True,
    )

    # Calculate the time taken for inference
    inference_time = time.time() - start_time

    # Log the time taken to 'time.txt'
    with open(TIME_FILE, "a") as time_file:
        time_file.write(f"{inference_time:.3f}\n")

    # Capture the prediction or error from the stdout of the subprocess
    prediction = result.stdout.strip()

    # If the internal script has an error, capture it
    if "Error:" in prediction:
        prediction = f"Error with {file_path}: {prediction}"

    # Log the result (prediction) to 'results.txt'
    with open(RESULTS_FILE, "a") as results_file:
        results_file.write(f"{prediction}\n")


def process_files_in_data_dir():
    # Iterate over all files in the data directory
    for filename in os.listdir(DATA_DIR):
        if filename.lower().endswith((".wav", ".mp3")):
            file_path = os.path.join(DATA_DIR, filename)
            run_internal_inference(file_path)


if __name__ == "__main__":
    process_files_in_data_dir()
    print("Inference complete! Time and results have been logged.")
