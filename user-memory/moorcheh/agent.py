import os
import re
from typing import Iterable, List

from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras
from moorcheh_sdk import MoorchehClient
from openai import OpenAI
import requests

load_dotenv()

MOORCHEH_API_KEY = os.getenv("MOORCHEH_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class ResearchMemoryAgent:
    """
    Retrieval-augmented agent backed by:
    - Moorcheh (vector memory store)
    - OpenAI embeddings (semantic vectors)
    - Cerebras (LLM generation)
    """

    def __init__(
        self,
        base_namespace: str = "my-vectors",
        embedding_model: str = "text-embedding-3-small",
        embedding_dimension: int = 1536,
    ) -> None:
        if not MOORCHEH_API_KEY:
            raise ValueError("MOORCHEH_API_KEY not found in environment variables.")
        if not CEREBRAS_API_KEY:
            raise ValueError("CEREBRAS_API_KEY not found in environment variables.")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        self.base_namespace = base_namespace
        self.namespace_name = f"{base_namespace}-{embedding_dimension}"
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.moorcheh = MoorchehClient(
            api_key=MOORCHEH_API_KEY,
            base_url="https://api.moorcheh.ai/v1",
        )
        self.cerebras = Cerebras(api_key=CEREBRAS_API_KEY)
        self.openai = OpenAI(api_key=OPENAI_API_KEY)

        self._ensure_namespace()

    def _ensure_namespace(self) -> None:
        try:
            self.moorcheh.create_namespace(
                namespace_name=self.namespace_name,
                type="vector",
                vector_dimension=self.embedding_dimension,
            )
        except Exception:
            # Namespace already exists or service returned a conflict; proceed safely.
            pass

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        response = self.openai.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

    def upsert_vectors(self, vectors: List[dict]) -> None:
        result = self.moorcheh.upload_vectors(
            namespace_name=self.namespace_name,
            vectors=vectors,
        )
        print(f"Uploaded {result} vectors")

    def ingest_texts(self, texts: List[str], source: str) -> None:
        embeddings = self.embed_texts(texts)
        vectors = [
            {
                "id": f"{source}-{idx}",
                "vector": embeddings[idx],
                "source": source,
                "index": idx,
                "text": texts[idx],
            }
            for idx in range(len(texts))
        ]
        self.upsert_vectors(vectors)

    def ingest_texts_batched(self, texts: List[str], source: str, batch_size: int = 64) -> None:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            self.ingest_texts(batch, source=f"{source}-batch-{start}")

    def retrieve(self, query_vector: List[float], top_k: int = 3) -> List[dict]:
        results = self.moorcheh.search(
            namespaces=[self.namespace_name],
            query=query_vector,
            top_k=top_k,
        )
        return results.get("results", [])

    def answer(self, question: str, top_k: int = 3) -> str:
        query_vector = self.embed_text(question)
        contexts = self.retrieve(query_vector, top_k=top_k)
        context_text = "\n".join([str(c) for c in contexts]) if contexts else "No context found."
        prompt = (
            "You are a research assistant. Use the context to answer precisely.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question:\n{question}\n"
        )

        completion = self.cerebras.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b",
            max_completion_tokens=1024,
            temperature=0.2,
            top_p=1,
            stream=False,
        )

        return completion.choices[0].message.content

    def log_turn(self, question: str, answer: str) -> None:
        self.ingest_texts([f"Q: {question}\nA: {answer}"], source="conversation")


def clean_text(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_url_texts(urls: Iterable[str]) -> List[str]:
    texts = []
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ResearchMemoryAgent/1.0; +https://example.com/bot)"
    }
    for url in urls:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code >= 400:
            fallback = f"https://r.jina.ai/http://{url.replace('https://', '').replace('http://', '')}"
            response = requests.get(fallback, headers=headers, timeout=15)
        response.raise_for_status()
        texts.append(clean_text(response.text))
    return texts


def ingest_urls_streaming(
    agent: "ResearchMemoryAgent",
    urls: Iterable[str],
    source_prefix: str,
    chunk_size: int = 1200,
    overlap: int = 150,
    batch_size: int = 16,
    max_chars_per_doc: int = 120_000,
) -> None:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ResearchMemoryAgent/1.0; +https://example.com/bot)"
    }
    buffer: List[str] = []
    total_chunks = 0
    for url in urls:
        print(f"Fetching: {url}")
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code >= 400:
            fallback = f"https://r.jina.ai/http://{url.replace('https://', '').replace('http://', '')}"
            response = requests.get(fallback, headers=headers, timeout=15)
        response.raise_for_status()
        text = clean_text(response.text)
        if max_chars_per_doc > 0:
            text = text[:max_chars_per_doc]
        for chunk in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            buffer.append(chunk)
            if len(buffer) >= batch_size:
                agent.ingest_texts(buffer, source=f"{source_prefix}-batch-{total_chunks}")
                total_chunks += len(buffer)
                buffer = []
        print(f"Done: {url}")
    if buffer:
        agent.ingest_texts(buffer, source=f"{source_prefix}-batch-{total_chunks}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 150) -> List[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap.")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


if __name__ == "__main__":
    agent = ResearchMemoryAgent()

    urls = [
        "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)",
        "https://en.wikipedia.org/wiki/Vector_database",
        "https://www.cerebras.net/blog/",
        "https://openai.com/research/",
        "https://en.wikipedia.org/wiki/Information_retrieval",
    ]

    ingest_urls_streaming(
        agent,
        urls,
        source_prefix="web-docs",
        chunk_size=1200,
        overlap=150,
        batch_size=16,
    )

    questions = [
        "Given the sources we just ingested, compare transformer attention with classic IR in how they handle relevance.",
        "Now propose a hybrid retrieval strategy that uses both sparse and dense signals, and explain when each dominates.",
        "Given your previous answer, how would you tune this strategy for a low-latency production API?",
        "Summarize the final plan as a set of engineering trade-offs with clear risks and mitigations.",
    ]

    for question in questions:
        response = agent.answer(question, top_k=6)
        print(f"\nQ: {question}\nA: {response}\n")
        agent.log_turn(question, response)