import torch

from config import SYSTEM_PROMPT
from generate_base import load_model_and_tokenizer


def main() -> None:
    model, tokenizer, device = load_model_and_tokenizer()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Explain dropout in one sentence."},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        model_output = model(**inputs)

    print("Input shape:", inputs["input_ids"].shape)
    print("Logits shape:", model_output.logits.shape)

    next_token_logits = model_output.logits[0, -1, :]
    next_token_probabilities = torch.softmax(next_token_logits, dim=-1)

    top_probabilities, top_token_ids = torch.topk(
        next_token_probabilities,
        k=10,
    )

    print("\nTop next-token candidates:\n")

    for rank, (probability, token_id) in enumerate(
        zip(top_probabilities, top_token_ids),
        start=1,
    ):
        token_text = tokenizer.decode([token_id.item()])

        print(
            f"{rank}. token={repr(token_text)}",
            f"id={token_id.item()}",
            f"probability={probability.item():.4f}",
        )


if __name__ == "__main__":
    main()
