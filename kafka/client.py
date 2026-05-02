from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, ConsumerRecord
import asyncio
import json
from s3.client.client import S3Client
from model.model import Audio2MusicInference

class KafkaClient:
    
    def __init__(
            self,
            consumer: AIOKafkaConsumer,
            s3_client: S3Client,
            producer: AIOKafkaProducer,
            inference: Audio2MusicInference
        ):
        self._consumer = consumer
        self._s3_client = s3_client
        self._producer = producer
        self._inference = inference
    
    async def process_message(self, message: ConsumerRecord):
        payload = json.loads(message.value.decode())
        await self._s3_client.get_object(payload['file_key'])
        output_midi, output_musicxml = self._inference.transcribe_audio(payload['file_key'])
        await self._s3_client.upload_file(output_musicxml)
        await self._producer.send(
            'almost-end-topic',
            json.dumps(
                {
                    "status": "complete",
                    "file_key": output_musicxml,
                }
            ).encode('utf-8')
        )

    async def process_queue(self,):
        async for message in self._consumer:
            asyncio.create_task(self.process_message(message))