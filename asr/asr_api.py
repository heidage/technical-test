import wave
import os
import aiofiles
import torch
import librosa

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

app = FastAPI(title="ASR API to transcribe audio files")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ASRResponse(BaseModel):
    transcription: str
    duration: str

@app.get("/ping")
def ping():
    return JSONResponse(content={"message": "pong"})

@app.post("/asr",response_model=ASRResponse)
async def trascribe_audio(file: UploadFile = File(...)):
    # check file extension ends with .mp3
    if not file.filename.endswith(".mp3"):
        return JSONResponse(content={"error": "File must be in mp3 format"}, status_code=422)
    
    # save file temporarily
    try:
        file_path = f"temp_{file.filename.split('/')[-1]}"
        async with aiofiles.open(file_path, "wb") as f:
            audio = await file.read()
            await f.write(audio)
        
        # downsample/ upsample audio to 16kHz
        audio_input, sr = librosa.load(file_path, sr=16000)
        
        # load model and processor before the server starts
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

        # pad input values and return pt tensor
        input_values = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding="longest").input_values

        # retrieve logits
        logits = model(input_values).logits

        # take argmax and decode
        predicited_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicited_ids)[0]

        print("Input values shape:", audio_input.shape, "sampling rate:", sr)

        return ASRResponse(transcription=transcription, duration=f"{librosa.get_duration(y=audio_input, sr=sr)}")
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    # delete file after the whole process
    finally:
        os.remove(file_path)