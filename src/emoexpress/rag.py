from typing import Any

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from emoexpress.config import (OPENAI_API_KEY,VECTOR_STORE_DIR)


EMBEDDING_MODEL = "text-embedding-3-small"


def load_vector_store() -> Chroma:
    """Load the existing Chroma database."""

    if not VECTOR_STORE_DIR.exists():
        raise FileNotFoundError("Chroma database was not found. Run the RAG notebook first.")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL,api_key=OPENAI_API_KEY)

    vector_store = Chroma(persist_directory=str(VECTOR_STORE_DIR),embedding_function=embeddings)

    if vector_store._collection.count() == 0:
        raise RuntimeError("The Chroma collection is empty.")

    return vector_store


def build_retrieval_query(user_story: str,emotion_predictions: list[dict],topic_result: dict) -> str:
    emotion_names = [prediction["emotion"] for prediction in emotion_predictions]

    predicted_topic = topic_result["topic"]

    return f"""
User situation:
{user_story}

Detected emotions:
{", ".join(emotion_names)}

Predicted topic:
{predicted_topic}

Retrieve safe, practical, non-diagnostic guidance,
coping strategies, and constructive next steps
relevant to this situation.
""".strip()


def retrieve_documents(vector_store: Chroma,query: str,predicted_topic: str,k: int = 5) -> tuple[list, str]:
    primary_results = (
        vector_store
        .similarity_search_with_relevance_scores(query=query,k=k,filter={"topic": predicted_topic}))

    if primary_results:
        return (primary_results,"predicted_topic")

    general_results = (vector_store.similarity_search_with_relevance_scores(query=query,k=k,filter={"topic": "general_support"}))

    if general_results:
        return (general_results,"general_support")

    broad_results = (vector_store.similarity_search_with_relevance_scores(query=query,k=k))

    return broad_results, "all_topics"


def format_retrieved_context(retrieved_results: list) -> tuple[str, list[dict]]:
    context_sections = []
    sources = []

    for rank, (document,relevance_score) in enumerate(retrieved_results,start=1):
        title = document.metadata.get("document_title","Unknown document")

        page = document.metadata.get("page_number","Unknown")

        topic = document.metadata.get("topic","Unknown")

        context_sections.append(
            f"""
SOURCE {rank}
Title: {title}
Page: {page}
Topic: {topic}

Passage:
{document.page_content}
""".strip()
        )

        sources.append({"rank": rank,
                "title": title,
                "page": page,
                "topic": topic,
                "source_file": (document.metadata.get("source_file")),"relevance_score": float(relevance_score)})

    return ("\n\n".join(context_sections),sources)