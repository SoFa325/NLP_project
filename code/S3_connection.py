import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from aiobotocore.session import get_session
from botocore.exceptions import ClientError


class S3Client:
    def __init__(
            self,
            access_key: str,
            secret_key: str,
            endpoint_url: str,
            bucket_name: str,
    ):
        self.config = {
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "endpoint_url": endpoint_url,
        }
        self.bucket_name = bucket_name
        self.session = get_session()

    @asynccontextmanager
    async def get_client(self):
        async with self.session.create_client("s3", **self.config) as client:
            yield client

    async def upload_file(
            self,
            file_path: str,
            s3_destination_path: str = None,
            create_path: bool = True
    ):
        try:
            object_name = s3_destination_path or Path(file_path).name

            if create_path:
                # Ensure path exists in S3 (create empty directory markers if needed)
                await self._ensure_path_exists(object_name)

            async with self.get_client() as client:
                with open(file_path, "rb") as file:
                    await client.put_object(
                        Bucket=self.bucket_name,
                        Key=object_name,
                        Body=file,
                    )
                print(f"File {file_path} uploaded to {self.bucket_name}/{object_name}")

        except ClientError as e:
            print(f"Error uploading file: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    async def _ensure_path_exists(self, s3_path: str):
        """Create directory structure in S3 if needed"""
        if "/" not in s3_path:
            return

        path_parts = s3_path.split("/")[:-1]  # Exclude filename
        current_path = ""

        async with self.get_client() as client:
            for part in path_parts:
                current_path = f"{current_path}{part}/" if current_path else f"{part}/"
                try:
                    await client.put_object(
                        Bucket=self.bucket_name,
                        Key=current_path,
                        Body=b'',
                    )
                except ClientError:
                    continue

    async def delete_file(self, object_name: str):
        try:
            async with self.get_client() as client:
                await client.delete_object(Bucket=self.bucket_name, Key=object_name)
                print(f"File {object_name} deleted from {self.bucket_name}")
        except ClientError as e:
            print(f"Error deleting file: {e}")

    async def get_file(self, object_name: str, destination_path: str):
        try:
            async with self.get_client() as client:
                response = await client.get_object(Bucket=self.bucket_name, Key=object_name)
                data = await response["Body"].read()
                with open(destination_path, "wb") as file:
                    file.write(data)
                print(f"File {object_name} downloaded to {destination_path}")
        except ClientError as e:
            print(f"Error downloading file: {e}")

"""
async def main():
    s3_client = S3Client(
        access_key="YCAJEI7og7KViaXXyhPIcFgvi",
        secret_key="YCNS3Qu3_9RKoOOXHgu7Y_amzVcLj5qE9S2mK4UA",
        endpoint_url="https://storage.yandexcloud.net",  # для Selectel используйте https://s3.storage.selcloud.ru
        bucket_name="nlpstorage",
    )

    # Проверка, что мы можем загрузить, скачать и удалить файл
    #await s3_client.upload_file("../PDFiles/Files/414.pdf")
    await s3_client.get_file("test.txt", "text_local_file.txt")
    #await s3_client.upload_file("test1.txt")
    #await s3_client.delete_file("414.pdf")

"""

"""
if __name__ == "__main__":
    asyncio.run(main())
"""
