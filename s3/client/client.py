from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import aiofiles  # type:ignore[import-untyped]
from aiobotocore.session import AioBaseClient, get_session
from botocore.exceptions import ClientError


class S3Client:
    """Клиент для взаимодействия с облачным хранилищем S3."""

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        endpoint_url: str,
        bucket_name: str,
    ) -> None:
        """Инициализация."""
        self.config = {
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "endpoint_url": endpoint_url,
        }
        self.bucket_name = bucket_name
        self.session = get_session()

    @asynccontextmanager
    async def _get_client(self) -> AsyncGenerator[Any, AioBaseClient]:
        """Получение клиента сессии."""
        async with self.session.create_client("s3", **self.config) as client:
            yield client

    async def upload_file(self, file_path: str) -> str:
        """Загрузить объект в хранилище S3. Возвращает ссылку на файл."""
        async with aiofiles.open(file_path, 'rb') as stream:
            data = await stream.read()
            async with self._get_client() as client:
                await client.put_object(
                    Bucket=self.bucket_name,
                    Key=file_path.split('/')[-1],
                    Body=data,
            )
        return file_path

    async def get_object(self, key: str) -> None:
        """Получить объект из хранилища S3."""
        async with self._get_client() as client:
            response = await client.get_object(Bucket=self.bucket_name, Key=key)
            async with response["Body"] as stream:
                data = await stream.read()
                async with aiofiles.open(key, "wb") as file:
                    await file.write(data)

    async def is_healthy(self) -> bool:
        """Проверка работоспособности Файловой БД."""
        try:
            async with self._get_client() as client:  # noqa:F841
                return True
        except ClientError:
            return False
