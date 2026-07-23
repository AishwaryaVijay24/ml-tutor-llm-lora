import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    LORA_ADAPTER_DIR,
    MODEL_ID,
    SYSTEM_PROMPT,
)


def load_chat_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)

    if LORA_ADAPTER_DIR.exists():
        print(f"Loading LoRA adapter from: {LORA_ADAPTER_DIR}")
        model = PeftModel.from_pretrained(
            base_model,
            str(LORA_ADAPTER_DIR),
        ).to(device)
    else:
        print("LoRA adapter not found. Using base model.")
        model = base_model

    model.eval()

    return model, tokenizer, device


def generate_chat_response(
    model,
    tokenizer,
    device: str,
    user_message: str,
    max_new_tokens: int = 220,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][input_length:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()


def main() -> None:
    model, tokenizer, device = load_chat_model()

    print("ML Tutor LLM Chat")
    print("Type `exit` to stop.")
    print(f"Device: {device}")

    while True:
        user_message = input("\nYou: ").strip()

        if user_message.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        if not user_message:
            continue

        response = generate_chat_response(
            model=model,
            tokenizer=tokenizer,
            device=device,
            user_message=user_message,
        )

        print("\nAssistant:")
        print(response)


if __name__ == "__main__":
    main()
