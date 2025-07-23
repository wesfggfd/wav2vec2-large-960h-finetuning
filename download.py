import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor
)
BEST_MODEL_PATH = "openai/whisper-large-v3"
teacher_model = WhisperForConditionalGeneration.from_pretrained(BEST_MODEL_PATH)
whisper_processor = WhisperProcessor.from_pretrained(BEST_MODEL_PATH)