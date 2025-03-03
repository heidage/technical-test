import requests
import pandas as pd
import os
from tqdm import tqdm

API_URL = "http://127.0.0.1:8001/asr"
CSV_PATH = "./cv-valid-dev.csv"
AUDIO_DIR = "C:/Users/shrey/Downloads/common_voice/cv-valid-dev"

def main():
    # load in csv first
    df = pd.read_csv(CSV_PATH)

    # create a list to store the transcriptions
    transcriptions = []

    # loop through the filenames in dataframe
    for filename in tqdm(df['filename'], desc="Transcribing"):
        # check if file exists and open 
        audio_path = os.path.join(AUDIO_DIR, filename)
        if os.path.exists(audio_path):
            with open(audio_path, "rb") as f:
                # send POST request to API endpoint
                response = requests.post(API_URL, files={"file":f})

            # check if request was successful
            if response.status_code == 200:
                # append the transcriptions to the list
                transcriptions.append(response.json()['transcription'])
            else:
                transcriptions.append("")

    # add the transcriptions to the dataframe
    df['generated_text'] = transcriptions

    # save new dataframe
    df.to_csv("cv-valid-dev.csv", index=False)

if __name__ == "__main__":
    main()