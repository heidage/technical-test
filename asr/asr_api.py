from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
    print(file)
    # Save the audio file
    with open("audio.wav", "wb") as audio:
        audio.write(await file.read())
    
    return ASRResponse(transcription="Hello World",duration="10s")