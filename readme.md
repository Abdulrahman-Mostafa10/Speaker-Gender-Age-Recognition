# 🎙️ Speaker Gender and Age Recognition
This project is a classical machine learning-based system that predicts speaker gender and age from audio recordings. It uses audio preprocessing, feature extraction (MFCC, chroma, spectral features), and models like KNN, LightGBM, XGBoost, and MLP.

# 🚀 Features
- Mono audio conversion and silence removal

- Rich audio feature extraction using Librosa

- Model inference for speaker gender and age classification

- Docker support for easy deployment

# 🔧 How to run
✅ Option 1: Run Locally via Git Clone
- Clone the Repository

  ```bash
  git clone https://github.com/your-username/speaker-age-gender-classification.git
  cd speaker-age-gender-classification
  ```

- Install Dependencies

  ```bash
  # For Linux/macOS:
  python3 -m venv venv
  source venv/bin/activate
  
  # For Windows:
  python -m venv venv
  venv\Scripts\activate

  pip install -r requirements.txt
  
  cd src/
  pip install -e . # this to package the project modules 
  ```
  

- Download the models from the Drive link above and place them into src/selected_model/

Put a `data` folder that contains the test audios inside `src` directory and then run `external_inference.py`   


# 🐳 Option 2: Run with Docker
- Create a folder “the base folder”.
  
- Change the directory to this folder.
  
- Put inside this folder the folder named “data” including the test audios.
  
- Open Docker Desktop.
  
- Run those two commands in the terminal “make sure to be in the base folder directory”.

  ```bash
  docker pull abdulrahmanmostafa/audio-inference:latest
  ```

  ```bash
  docker run --rm \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/output:/app/output" \
    abdulrahmanmostafa/audio-inference:latest
  ```
 
- Now you will find a directory named output the has the two text files `results.txt` and `time.txt`.
