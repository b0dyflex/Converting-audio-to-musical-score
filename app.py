from fastapi import FastAPI
from fastapi.responses import FileResponse
from model.model import Audio2MusicInference
import torch


app = FastAPI()

@app.post('/note')
def transcribe(audio_path: str):
    """Транскрибирование аудио в нотную партитуру."""
    model = Audio2MusicInference(
        model_path='/root/Converting-audio-to-musical-score/audio2music_epoch_15.pth',
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.transcribe_audio(audio_path=audio_path)
    return FileResponse('output.mid')