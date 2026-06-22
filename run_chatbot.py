import os

from dotenv import load_dotenv
from google import genai

from embedding import embed_texts
from query import interactive_loop
from vectordb import load_qdrant, search_qdrant

load_dotenv()

class GeminiModel:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def generate_content(self, prompt: str):
        class _R:
            def __init__(self, text):
                self.text = text

        response = self.client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
        )
        return _R(response.text)


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Thiếu GEMINI_API_KEY. Hãy sao chép .env.example thành .env "
            "và điền API key."
        )

    qdrant_client = load_qdrant()

    class QdrantStore:
        def search(self, question, top_k=5):
            query_vector = embed_texts([question])[0]
            return search_qdrant(query_vector, qdrant_client, top_k=top_k)

    interactive_loop(
        vector_store=QdrantStore(),
        gemini_model=GeminiModel(api_key),
    )


if __name__ == "__main__":
    main()
