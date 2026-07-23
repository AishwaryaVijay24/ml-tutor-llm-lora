from generate_base import generate_response, load_model_and_tokenizer


def main() -> None:
    model, tokenizer, device = load_model_and_tokenizer()

    prompt = "Explain dropout in neural networks with a simple example."

    experiments = [
        {
            "label": "A. Greedy decoding",
            "do_sample": False,
            "temperature": None,
            "top_p": None,
        },
        {
            "label": "B. Low temperature sampling: temperature=0.3, top_p=0.9",
            "do_sample": True,
            "temperature": 0.3,
            "top_p": 0.9,
        },
        {
            "label": "C. Balanced sampling: temperature=0.7, top_p=0.9",
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        {
            "label": "D. High temperature sampling: temperature=1.2, top_p=0.95",
            "do_sample": True,
            "temperature": 1.2,
            "top_p": 0.95,
        },
    ]

    print("Prompt:")
    print(prompt)

    for experiment in experiments:
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            device=device,
            user_message=prompt,
            max_new_tokens=200,
            do_sample=experiment["do_sample"],
            temperature=experiment["temperature"],
            top_p=experiment["top_p"],
        )

        print("=" * 80)
        print(experiment["label"])
        print("=" * 80)
        print(response)
        print()


if __name__ == "__main__":
    main()
