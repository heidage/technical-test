# technical-test

This README.md contains the information, steps taken for each task and the assumptions made while doing the task.

## Task 1:

I added the README.md, LICENSE and .gitignore files in the root directory. While the task specified to create a ```requirements.txt```, I created it and movede it to the ```asr``` folder for convenience.

## Task 2:
Under the folder ```asr```, these are the files inside it:
- ```asr_api.py```: contains the FastAPI endpoint for ```/asr```
- ```cv-decode.py```: contains the code to to call the API in ```asr_api.py``` and trasncribe the audio data under ```cv-valid-dev``` folder. The trascriptions are then saved to ```cv-valid-dev.csv```
- ```Dockerfile```: contains the build information to containerise asr_api.py
- ```requirements.txt```: contains the necessary packages in order to run the API and ```cv-decode.py```

### Run ```/asr``` endpoint
1. cd to ```/asr``` folder
2. run the commands: ```docker build -t asr-api .``` and ```docker run -p 8001:8001 asr-api```

### Run ```cv-decode.py```
1. Download the [common voice] (https://www.dropbox.com/scl/fi/i9yvfqpf7p8uye5o8k1sj/common_voice.zip?rlkey=lz3dtjuhekc3xw4jnoeoqy5yu&dl=0) dataset
2. Unzip and store it in your local directory
3. Create a virtual environment and use the ```requirements.txt``` in the ```asr``` folder
4. run the command ```python3 cv-decode.py```

**Assumptions made**
1. Since the audio being fed to ```wav2vec2-large-960h``` is required to be sampled at 16kHz, I used the ```librosa``` library to sample the audio at 16kHz when loading in using the audio file path. Hence I assumed that the audio should not be discarded but instead sampled at 16kHz
2. Assumed that the ```cv-decode.py``` should just use my local directory to the ```cv-valid-dev``` to fetch the audio. Therefore, I did not add it into the repository as the dataset is huge.