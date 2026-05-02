import os
import sys
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
from dependency_injector import containers, providers, resources
from dotenv import load_dotenv
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from kafka.client import KafkaClient
from model.model import Audio2MusicInference
from s3.client.client import S3Client
import torch

load_dotenv()

class ApplicationContainer(containers.DeclarativeContainer):
    """Контейнер с различными зависимостями приложения."""

    wiring_config = containers.WiringConfiguration(packages=['app'])

    s3_client = providers.Resource(
        S3Client,
        access_key=os.getenv("ACCESS_KEY"),
        secret_key=os.getenv("SECRET_KEY"),
        endpoint_url=os.getenv("ENDPOINT_URL"),
        bucket_name=os.getenv("BUCKET_NAME"),
    )

    def _init_audio_to_music_inference():
<<<<<<< HEAD
        sys.path.insert(0, str(Path.cwd()) + '/model')
=======
        sys.path.insert(0, str(Path.cwd().parent) + '/model')
>>>>>>> db8eeaeeb8b4391230bee846a25c2de1e8931d8b
        spec = spec_from_file_location("model", str(Path.cwd()) + '/model/model.py')
        module = module_from_spec(spec)
        sys.modules["model"] = module
        spec.loader.exec_module(module)
        model = Audio2MusicInference(
            model_path='../models/audio2music_best.pth',
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        return model


    audio_to_music_inference = providers.Resource(_init_audio_to_music_inference)
    _producer = providers.Resource(AIOKafkaProducer, bootstrap_servers='localhost:9092')
    _consumer = providers.Resource(AIOKafkaConsumer, 'middle-topic', bootstrap_servers='localhost:9092')
    kafka_client = providers.Resource(KafkaClient, _consumer, s3_client, _producer, audio_to_music_inference)