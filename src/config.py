from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

INSTRUCTION_DATA_PATH = DATA_DIR / "instruction_response_data.jsonl"
CHAT_DATA_PATH = DATA_DIR / "chat_instruction_data.jsonl"

PROCESSED_DATA_DIR = OUTPUT_DIR / "processed_dataset"
LORA_ADAPTER_DIR = OUTPUT_DIR / "lora_adapter"

SYSTEM_PROMPT = (
    "You are a beginner-friendly machine learning tutor. "
    "Explain clearly, use simple examples, and include a memory hook when useful."
)

TEST_PROMPTS = [
    "Explain overfitting with a simple example.",
    "What is a tokenizer in LLMs?",
    "Explain temperature and top-p simply.",
    "What is LoRA fine-tuning?",
]
