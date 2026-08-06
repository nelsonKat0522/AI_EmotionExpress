import json
from typing import Any

from openai import OpenAI

from emoexpress.config import OPENAI_API_KEY


RESPONSE_MODEL = "gpt-4o-mini"

client = OpenAI(api_key=OPENAI_API_KEY)


RESPONSE_SYSTEM_PROMPT = """
You are EmoExpress, an empathetic emotional-support assistant.

Use the user story, predicted emotions, predicted topic, and retrieved
PDF knowledge to provide safe and practical emotional support.

Rules:

1. Do not diagnose medical or mental-health conditions.
2. Do not recommend medication changes.
3. Do not claim to be a therapist, physician, or crisis professional.
4. Do not invent document titles, page numbers, or citations.
5. Use retrieved sources only when they directly support a recommendation.
6. Keep recommendations realistic, practical, and manageable.
7. Generate two or three recommendations.
8. The caption must contain no more than 15 words.
9. The image prompt should represent emotional progress and realistic hope.
10. Do not include graphic distress, violence, self-harm, medication,
    medical treatment, logos, interface elements, or written words
    inside the generated scene.
11. Return valid JSON only.

Required JSON format:

{
    "empathetic_response": "string",
    "recommendations": [
        {
            "recommendation": "string",
            "source_title": "string or null",
            "page": "integer or null"
        }
    ],
    "caption": "string",
    "image_prompt": "string",
    "retrieval_status": "grounded, partially_grounded, or not_grounded"
}
""".strip()


def build_response_prompt(user_story: str,emotion_predictions: list[dict],
                        topic_result: dict,retrieved_context: str,retrieved_sources: list[dict]) -> str:
    """Build the grounded response-generation prompt."""

    return f"""
USER STORY

{user_story}

EMOTION PREDICTIONS

{json.dumps(emotion_predictions,indent=2,ensure_ascii=False)}

PREDICTED KNOWLEDGE-BASE TOPIC

{json.dumps(topic_result,indent=2,ensure_ascii=False)}

RETRIEVED KNOWLEDGE

{retrieved_context}

AVAILABLE SOURCES

{json.dumps(retrieved_sources,indent=2,ensure_ascii=False)}

Generate:

1. A short empathetic acknowledgment of approximately 2–4 sentences.
2. Two or three practical recommendations.
3. A short encouraging caption of no more than 15 words.
4. A safe image-generation prompt.
5. Citations only when supported by the retrieved passages.

Return valid JSON only.
""".strip()


def validate_generated_response(response_data: dict) -> dict:
    """Validate and normalize the generated response."""

    required_fields = {"empathetic_response","recommendations","caption","image_prompt","retrieval_status"}

    missing_fields = (required_fields - set(response_data.keys()))

    if missing_fields:
        raise ValueError(f"Generated response is missing fields: {missing_fields}")

    recommendations = response_data["recommendations"]

    if not isinstance(recommendations, list):
        raise TypeError("recommendations must be a list.")

    normalized_recommendations = []

    for item in recommendations:
        if not isinstance(item, dict):
            continue

        recommendation_text = str(
            item.get("recommendation","")).strip()

        if not recommendation_text:
            continue

        normalized_recommendations.append({"recommendation": recommendation_text,
                "source_title": item.get("source_title"),"page": item.get("page")})

    valid_statuses = {"grounded","partially_grounded","not_grounded"}

    retrieval_status = response_data.get("retrieval_status","partially_grounded")

    if retrieval_status not in valid_statuses:
        retrieval_status = ("partially_grounded")

    return {"empathetic_response": str(response_data["empathetic_response"]).strip(),"recommendations": (normalized_recommendations),
        "caption": str(response_data["caption"]).strip(),"image_prompt": str(response_data["image_prompt"]).strip(),"retrieval_status": retrieval_status}


def generate_grounded_response(user_story: str,emotion_predictions: list[dict],topic_result: dict,retrieved_context: str,
                               retrieved_sources: list[dict],model: str = RESPONSE_MODEL) -> dict:
    """Generate the grounded response, caption, and image prompt."""

    prompt = build_response_prompt(user_story=user_story,
                                    emotion_predictions=emotion_predictions,
                                    topic_result=topic_result,
                                    retrieved_context=retrieved_context,
                                    retrieved_sources=retrieved_sources)

    response = client.chat.completions.create(model=model,
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system","content": (RESPONSE_SYSTEM_PROMPT)},{"role": "user","content": prompt}])

    raw_content = (response.choices[0].message.content)

    if not raw_content:
        raise RuntimeError("The response model returned empty content.")

    parsed_response = json.loads(raw_content)

    validated_response = (validate_generated_response(parsed_response))

    validated_response["model"] = model

    return validated_response