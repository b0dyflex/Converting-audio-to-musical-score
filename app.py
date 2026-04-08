from fastapi import FastAPI
from fastapi.responses import FileResponse
from model.model import Audio2MusicInference
<<<<<<< HEAD
from containers import ApplicationContainer
=======
>>>>>>> b234dd04b347751390542835d35f6c2a3fd1fff2
import torch
from importlib.util import spec_from_file_location, module_from_spec
import os
import sys
<<<<<<< HEAD
from contextlib import asynccontextmanager
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    container = ApplicationContainer()
    container.init_resources()
    consumer = await container.kafka_client()
    asyncio.create_task(consumer.process_queue())
    yield
    container.shutdown_resources()


app = FastAPI(lifespan=lifespan)
=======

app = FastAPI()
>>>>>>> b234dd04b347751390542835d35f6c2a3fd1fff2

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