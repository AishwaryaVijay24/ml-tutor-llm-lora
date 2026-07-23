import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    LORA_ADAPTER_DIR,
    MODEL_ID,
    OUTPUT_DIR,
    SYSTEM_PROMPT,
    TEST_PROMPTS,
)
from generate_base import generate_response


def load_base_model(tokenizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    base_model.eval()
    return base_model, device


def load_tuned_model():
    if not LORA_ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found at {LORA_ADAPTER_DIR}. "
            "Run `python src/train_lora.py` first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    tuned_model = PeftModel.from_pretrained(
        base_model,
        str(LORA_ADAPTER_DIR),
    ).to(device)

    tuned_model.eval()

    return tuned_model, tokenizer, device


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    base_model, device = load_base_model(tokenizer)
    tuned_model, tuned_tokenizer, tuned_device = load_tuned_model()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "evaluation_results.md"

    sections = ["# Base Model vs LoRA-Tuned Model Evaluation\n"]

    for prompt in TEST_PROMPTS:
        base_response = generate_response(
            model=base_model,
            tokenizer=tokenizer,
            device=device,
            user_message=prompt,
            max_new_tokens=200,
        )

        tuned_response = generate_response(
            model=tuned_model,
            tokenizer=tuned_tokenizer,
            device=tuned_device,
            user_message=prompt,
            max_new_tokens=200,
        )

        sections.append(f"## Prompt\n\n{prompt}\n")
        sections.append("### Base model response\n")
        sections.append(base_response + "\n")
        sections.append("### LoRA-tuned response\n")
        sections.append(tuned_response + "\n")

    output_path.write_text("\n".join(sections), encoding="utf-8")

    print(f"Evaluation saved to: {output_path}")


if __name__ == "__main__":
    main()
