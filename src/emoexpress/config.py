from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"

EMOTION_MODEL_NAME = ("SamLowe/roberta-base-go_emotions")

TOPIC_MODEL_DIR = (MODEL_DIR/ "topic_classifier"/ "distilbert_topic_classifier")

KNOWLEDGE_BASE_DIR = (PROJECT_ROOT/ "knowledge_base")

VECTOR_STORE_DIR = (PROJECT_ROOT/ "vector_store"/ "chroma_db")

OUTPUT_DIR = PROJECT_ROOT / "outputs"

GENERATED_IMAGE_DIR = (OUTPUT_DIR/ "generated_images")

GENERATED_RESPONSE_DIR = (OUTPUT_DIR/ "generated_responses")

GENERATED_IMAGE_DIR.mkdir(parents=True,exist_ok=True)

GENERATED_RESPONSE_DIR.mkdir(parents=True,exist_ok=True)

HISTORY_DIR = (OUTPUT_DIR/ "history")

HISTORY_FILE = (HISTORY_DIR/ "emoexpress_history.json")

HISTORY_DIR.mkdir(parents=True,exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY was not found in the .env file.")