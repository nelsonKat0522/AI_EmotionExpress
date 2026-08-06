from typing import Any

import torch
from transformers import (AutoModelForSequenceClassification,AutoTokenizer)

from emoexpress.config import (KNOWLEDGE_BASE_DIR,TOPIC_MODEL_DIR)


MAX_TOPIC_LENGTH = 128


def load_topic_resources() -> tuple[Any, Any]:
    """Load the fine-tuned DistilBERT topic classifier."""

    if not TOPIC_MODEL_DIR.exists():
        raise FileNotFoundError(f"Topic model not found: {TOPIC_MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(str(TOPIC_MODEL_DIR))

    model = (AutoModelForSequenceClassification.from_pretrained(str(TOPIC_MODEL_DIR)))

    model.eval()

    return tokenizer, model


def get_knowledge_base_topics() -> set[str]:
    """Return valid topic-folder names."""

    if not KNOWLEDGE_BASE_DIR.exists():
        return {"general_support"}

    return {directory.name
        for directory
        in KNOWLEDGE_BASE_DIR.iterdir()
        if directory.is_dir()}


def predict_topic(user_story: str,tokenizer: Any,model: Any,device: torch.device) -> dict:
    """Predict one knowledge-base topic."""

    if not isinstance(user_story, str):
        raise TypeError("user_story must be a string.")

    user_story = user_story.strip()

    if not user_story:
        raise ValueError("The user story cannot be empty.")

    model.to(device)

    encoded_input = tokenizer(user_story,return_tensors="pt",truncation=True,max_length=MAX_TOPIC_LENGTH,padding=True)

    encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

    with torch.no_grad():
        output = model(**encoded_input)

    probabilities = torch.softmax(output.logits,dim=-1)[0]

    predicted_id = int(
        torch.argmax(probabilities).item())

    predicted_topic = (model.config.id2label[predicted_id])

    valid_topics = get_knowledge_base_topics()

    if predicted_topic not in valid_topics:
        predicted_topic = "general_support"

    return {"topic": predicted_topic,
        "confidence": float(probabilities[predicted_id].item()),
        "model": "Fine-Tuned DistilBERT"}