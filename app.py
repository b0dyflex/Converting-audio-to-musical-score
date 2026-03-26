from fastapi import FastAPI
from fastapi.responses import FileResponse
from model.model import Audio2MusicInference
import torch
from importlib.util import spec_from_file_location, module_from_spec
import os
import sys

app = FastAPI()

@app.post('/note')
def transcribe(audio_path: str):
    """Транскрибирование аудио в нотную партитуру."""
    sys.path.insert(0, '/root/Converting-audio-to-musical-score/model')
    spec = spec_from_file_location("model", '/root/Converting-audio-to-musical-score/model/model.py')
    module = module_from_spec(spec)
    sys.modules["model"] = module
    spec.loader.exec_module(module)
    model = Audio2MusicInference(
        model_path='../models/audio2music_best.pth',
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.transcribe_audio(audio_path=audio_path)
    return FileResponse('output.mid')