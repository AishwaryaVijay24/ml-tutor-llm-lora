import torch
from datasets import load_from_disk
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from config import (
    LORA_ADAPTER_DIR,
    MODEL_ID,
    PROCESSED_DATA_DIR,
)


def main() -> None:
    if not PROCESSED_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {PROCESSED_DATA_DIR}. "
            "Run `python src/prepare_dataset.py` first."
        )

    dataset = load_from_disk(str(PROCESSED_DATA_DIR))

    print(dataset)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    training_args = SFTConfig(
        output_dir=str(LORA_ADAPTER_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        max_length=512,
        packing=False,
        report_to=[],
        model_init_kwargs={
            "torch_dtype": dtype,
        },
    )

    trainer = SFTTrainer(
        model=MODEL_ID,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
    )

    trainer.train()

    trainer.save_model(str(LORA_ADAPTER_DIR))

    print(f"LoRA adapter saved to: {LORA_ADAPTER_DIR}")


if __name__ == "__main__":
    main()
