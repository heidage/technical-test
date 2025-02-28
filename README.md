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

## Task 3
Under the folder asr-train, there is a file `cv-train-2a.ipynb` which contains the code used to fine-tune the `wav2vec2-large-960h` model. The fine tuned model is saved and the folder name is `wav2vec2-large-960h-cv`.

All the code explanations for chosen preprocessing, tokenizer, feature extraction and pipeline processes are written in the ipynb file.

There is `vocab.json` file which consists of the vocabulary saved for the tokenizer and the `cv-valid-train.csv` was used for fine tuning and evaluating the model with a 70-30 split.

The overall performance, where it transcribed the common-voice mp3 files under cv-valid-test folder and compared againt the the text in `cv-valid-test.csv` is stored in a seperate csv called `performance.csv`

## Task 4
The answers for this task is written in `training-report.pdf`.

## Task 5
All the necesarry code files: `cv-hotword-5a.ipynb` and `cv-hotword-similarity-5b.ipynb` is inside the folder `hotword-detection`. For task 5a, the file `detected.txt` is saved and shows the list of mp3 filenames together with the hot words detected in each file. 


You can use the file `cv-valid-dev.csv` which is in the folder when you need to import the dataset. It also contains the column `similarity`, which contains `True` or `False` values.



## Task 6
The essay for this task is written in `essay-ssl.pdf`