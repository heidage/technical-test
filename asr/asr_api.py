import wave
import os
import aiofiles

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tempfile import NamedTemporaryFile

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
    duration:str

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
        file_path = f"temp_{file.filename.split("/")[-1]}"
        async with aiofiles.open(file_path, "wb") as f:
            audio = await file.read()
            await f.write(audio)
        
        # check if sampling frequency is 16 KHz
        with wave.open(file_path) as f:
            if f.getframerate() != 16000:
                return JSONResponse(content={"error": "Sampling frequency must be 16 KHz"}, status_code=422)
            
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    # delete file after the whole process
    finally:
        os.remove(file_path)
    return ASRResponse(transcription="Hello World",duration="10s")