from typing import Any

import numpy as np
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

from emoexpress.config import EMOTION_MODEL_NAME

EMOTION_THRESHOLD = 0.50
MAXIMUM_EMOTIONS = 3
MAX_EMOTION_LENGTH = 128


def load_emotion_resources() -> tuple[Any, Any, torch.device]:
    """Load the pretrained RoBERTa emotion classifier."""

    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_NAME)

    model = (AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_NAME))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    return tokenizer, model, device


def get_emotion_labels(model: Any) -> list[str]:
    """Return emotion names in model-output order."""

    id_to_label = model.config.id2label

    return [id_to_label[index] for index in range(len(id_to_label))]


def predict_emotions(user_story: str,tokenizer: Any,model: Any,device: torch.device, threshold: float = EMOTION_THRESHOLD,
                      maximum_emotions: int = MAXIMUM_EMOTIONS) -> list[dict]:
    """Predict multiple emotions from one user story."""

    if not isinstance(user_story, str):
        raise TypeError("user_story must be a string.")

    user_story = user_story.strip()

    if not user_story:
        raise ValueError("The user story cannot be empty.")

    encoded_input = tokenizer(user_story,return_tensors="pt",truncation=True,max_length=MAX_EMOTION_LENGTH,padding=True)

    encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

    with torch.no_grad():
        output = model(**encoded_input)

    probabilities = (torch.sigmoid(output.logits)[0].cpu().numpy())

    emotion_labels = get_emotion_labels(model)

    selected_indices = np.where(probabilities >= threshold)[0]

    if len(selected_indices) == 0:
        selected_indices = np.array([int(np.argmax(probabilities))])

    selected_indices = sorted(selected_indices,key=lambda index: probabilities[index],reverse=True)[:maximum_emotions]

    return [{"emotion": emotion_labels[index],
            "score": float(probabilities[index])} for index in selected_indices]