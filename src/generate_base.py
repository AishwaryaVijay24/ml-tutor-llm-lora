import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_ID, SYSTEM_PROMPT


def load_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    model.eval()

    return model, tokenizer, device


def generate_response(
    model,
    tokenizer,
    device: str,
    user_message: str,
    max_new_tokens: int = 180,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
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

    generation_args = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        generation_args["temperature"] = temperature
        generation_args["top_p"] = top_p

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_args,
        )

    input_length = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][input_length:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()


def main() -> None:
    model, tokenizer, device = load_model_and_tokenizer()
    print("Using device:", device)

    prompt = "Explain overfitting with a simple example."
    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        device=device,
        user_message=prompt,
        max_new_tokens=220,
    )

    print("\nPrompt:")
    print(prompt)
    print("\nBase model response:")
    print(response)


if __name__ == "__main__":
    main()
