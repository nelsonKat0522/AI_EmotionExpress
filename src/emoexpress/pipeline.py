import time
from typing import Any

from emoexpress.emotion_classifier import (
    predict_emotions,
)
from emoexpress.topic_classifier import (
    predict_topic,
)
from emoexpress.rag import (
    build_retrieval_query,
    format_retrieved_context,
    retrieve_documents,
)
from emoexpress.response_generator import (
    generate_grounded_response,
)
from emoexpress.image_generator import (
    generate_image_safely,
)


def run_emoexpress_pipeline(
    user_story: str,
    emotion_tokenizer: Any,
    emotion_model: Any,
    topic_tokenizer: Any,
    topic_model: Any,
    device: Any,
    vector_store: Any,
    retrieval_k: int = 5,
    generate_image_output: bool = True,
) -> dict:
    """Run the complete EmoExpress inference pipeline."""

    pipeline_start = time.perf_counter()

    if not isinstance(user_story, str):
        raise TypeError(
            "user_story must be a string."
        )

    user_story = user_story.strip()

    if len(user_story) < 5:
        raise ValueError(
            "Please enter a more detailed story."
        )

    # Emotion classification
    emotion_start = time.perf_counter()

    emotion_predictions = predict_emotions(
        user_story=user_story,
        tokenizer=emotion_tokenizer,
        model=emotion_model,
        device=device,
    )

    emotion_time = (
        time.perf_counter()
        - emotion_start
    )

    # Topic classification
    topic_start = time.perf_counter()

    topic_result = predict_topic(
        user_story=user_story,
        tokenizer=topic_tokenizer,
        model=topic_model,
        device=device,
    )

    topic_time = (
        time.perf_counter()
        - topic_start
    )

    predicted_topic = topic_result[
        "topic"
    ]

    # Build retrieval query
    retrieval_query = build_retrieval_query(
        user_story=user_story,
        emotion_predictions=emotion_predictions,
        topic_result=topic_result,
    )

    # Retrieve knowledge
    retrieval_start = time.perf_counter()

    retrieved_results, strategy = (
        retrieve_documents(
            vector_store=vector_store,
            query=retrieval_query,
            predicted_topic=predicted_topic,
            k=retrieval_k,
        )
    )

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )

    retrieved_context, sources = (
        format_retrieved_context(
            retrieved_results
        )
    )

    if not retrieved_context.strip():
        retrieved_context = (
            "No directly relevant knowledge-base "
            "passages were found."
        )
        sources = []

    # Generate response
    response_start = time.perf_counter()

    generated_response = (
        generate_grounded_response(
            user_story=user_story,
            emotion_predictions=emotion_predictions,
            topic_result=topic_result,
            retrieved_context=retrieved_context,
            retrieved_sources=sources,
        )
    )

    response_time = (
        time.perf_counter()
        - response_start
    )

    # Generate image
    image_start = time.perf_counter()

    image_result = {
        "status": "skipped",
        "path": None,
        "final_path": None,
        "error": None,
    }

    if generate_image_output:
        image_result = (
            generate_image_safely(
                image_prompt=(
                    generated_response[
                        "image_prompt"
                    ]
                ),
                caption=(
                    generated_response[
                        "caption"
                    ]
                ),
            )
        )

    image_time = (
        time.perf_counter()
        - image_start
    )

    total_time = (
        time.perf_counter()
        - pipeline_start
    )

    return {
        "user_story": user_story,
        "emotion_predictions": (
            emotion_predictions
        ),
        "topic_result": topic_result,
        "retrieval": {
            "query": retrieval_query,
            "predicted_topic": (
                predicted_topic
            ),
            "strategy": strategy,
            "sources": sources,
        },
        "generated_response": (
            generated_response
        ),
        "generated_image": (
            image_result
        ),
        "timing": {
            "emotion_seconds": emotion_time,
            "topic_seconds": topic_time,
            "retrieval_seconds": retrieval_time,
            "response_seconds": response_time,
            "image_seconds": image_time,
            "total_seconds": total_time,
        },
    }