import json
from pathlib import Path

from datasets import load_dataset

from config import (
    CHAT_DATA_PATH,
    INSTRUCTION_DATA_PATH,
    PROCESSED_DATA_DIR,
)


def validate_instruction_row(row: dict) -> bool:
    instruction = str(row.get("instruction", "")).strip()
    response = str(row.get("response", "")).strip()

    if not instruction:
        return False

    if not response:
        return False

    if len(response.split()) < 8:
        return False

    return True


def convert_to_messages(example: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ]
    }


def main() -> None:
    print(f"Loading: {INSTRUCTION_DATA_PATH}")

    dataset = load_dataset(
        "json",
        data_files=str(INSTRUCTION_DATA_PATH),
        split="train",
    )

    print(dataset)
    print("First row:")
    print(dataset[0])

    valid_rows = []
    invalid_count = 0

    for row in dataset:
        if validate_instruction_row(row):
            valid_rows.append(row)
        else:
            invalid_count += 1

    print(f"Valid rows: {len(valid_rows)}")
    print(f"Invalid rows: {invalid_count}")

    CHAT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CHAT_DATA_PATH, "w", encoding="utf-8") as file:
        for row in valid_rows:
            chat_row = convert_to_messages(row)
            file.write(json.dumps(chat_row, ensure_ascii=False) + "\n")

    print(f"Saved chat dataset to: {CHAT_DATA_PATH}")

    chat_dataset = load_dataset(
        "json",
        data_files=str(CHAT_DATA_PATH),
        split="train",
    )

    split_dataset = chat_dataset.train_test_split(
        test_size=0.3,
        seed=42,
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    split_dataset.save_to_disk(str(PROCESSED_DATA_DIR))

    print(f"Saved processed dataset to: {PROCESSED_DATA_DIR}")
    print(split_dataset)


if __name__ == "__main__":
    main()
