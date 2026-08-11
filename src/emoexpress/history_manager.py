from datetime import datetime
from pathlib import Path
from typing import Any
import json
import uuid

from emoexpress.config import HISTORY_FILE


def load_history() -> list[dict[str, Any]]:
    """Load all saved EmoExpress stories."""

    if not HISTORY_FILE.exists():
        return []

    try:
        with open(HISTORY_FILE,"r",encoding="utf-8") as file:
            history = json.load(file)

        if not isinstance(history, list):
            return []

        return history

    except (json.JSONDecodeError,OSError):
        return []


def save_history(history: list[dict[str, Any]]) -> None:
    """Save all story records to the history JSON file."""

    HISTORY_FILE.parent.mkdir(parents=True,exist_ok=True)

    temporary_path = HISTORY_FILE.with_suffix(".tmp")

    with open(temporary_path,"w",encoding="utf-8") as file:
        json.dump(history,file,ensure_ascii=False,indent=4)

    temporary_path.replace(HISTORY_FILE)


def create_history_record(pipeline_result: dict[str, Any]) -> dict[str, Any]:
    """Convert a pipeline result into one history record."""

    generated_response = pipeline_result.get("generated_response",{})

    generated_image = pipeline_result.get("generated_image",{})

    topic_result = pipeline_result.get("topic_result",{})

    return {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "user_story": pipeline_result.get("user_story",""),
        "emotion_predictions": pipeline_result.get("emotion_predictions",[]),
        "topic": topic_result.get("topic","general_support"),
        "topic_confidence": topic_result.get("confidence",0.0),
        "empathetic_response": generated_response.get("empathetic_response",""),
        "recommendations": generated_response.get("recommendations",[]),
        "caption": generated_response.get("caption",""),
        "retrieval_status": generated_response.get("retrieval_status",""),
        "sources": pipeline_result.get("retrieval",{}).get("sources",[]),
        "image_path": (generated_image.get("final_path") or generated_image.get("path")),
        "timing": pipeline_result.get("timing",{}),}

def add_history_record(pipeline_result: dict[str, Any],) -> dict[str, Any]:
    """Add a pipeline result to the beginning of history."""

    history = load_history()

    record = create_history_record(pipeline_result)

    history.insert(0,record)

    save_history(history)

    return record


def delete_history_record(record_id: str,) -> bool:
    """Delete one record by ID."""

    history = load_history()

    updated_history = [record for record in history if record.get("id") != record_id]

    if len(updated_history) == len(history):
        return False

    save_history(updated_history)

    return True


def clear_history() -> None:
    """Remove all history records."""

    save_history([])