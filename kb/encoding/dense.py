from __future__ import annotations

from openai import AsyncOpenAI

from kb.config import Config


class DenseEncoder:
    def __init__(self, client: AsyncOpenAI, config: Config) -> None:
        self._client = client
        self._model = config.embedding_model
        self._dimensions = config.embedding_dimensions

    async def encode(self, text: str) -> list[float]:
        return await self.encode_batch([text])[0]

    async def encode_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimensions,
        )
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
