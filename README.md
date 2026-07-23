# ML Tutor LLM with LoRA

A small Hugging Face project that fine-tunes a compact instruction model to answer beginner-friendly machine learning questions.

This project demonstrates the practical LLM workflow:

```text
instruction-response data
   ↓
chat-format dataset
   ↓
tokenizer + small instruct model
   ↓
base model inference
   ↓
SFT + LoRA fine-tuning
   ↓
base vs tuned evaluation
   ↓
save/reload adapter
   ↓
simple chat CLI
```

## Project goals

- Understand how tokenizer/model inference works.
- Create clean instruction-response data.
- Fine-tune a compact instruct model using SFT + LoRA.
- Compare base model responses with tuned model responses.
- Save and reload the LoRA adapter.
- Build a small command-line chat app.

## Model

Default model:

```text
HuggingFaceTB/SmolLM2-360M-Instruct
```

It is intentionally small enough for learning experiments.

## Repository structure

```text
ml-tutor-llm-lora/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── instruction_response_data.jsonl
│   └── chat_instruction_data.jsonl
│
├── src/
│   ├── config.py
│   ├── prepare_dataset.py
│   ├── generate_base.py
│   ├── compare_decoding.py
│   ├── inspect_logits.py
│   ├── train_lora.py
│   ├── evaluate.py
│   └── chat_cli.py
│
└── outputs/
    └── sample_responses.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

In Google Colab, run:

```python
!pip install -r requirements.txt
```

If you are doing text-only LLM work and see a `torchvision::nms` error, try:

```python
!pip uninstall -y torchvision
```

Then restart the runtime.

## 1. Prepare dataset

```bash
python src/prepare_dataset.py
```

This validates the JSONL data and creates a processed train/test split.

## 2. Run base model inference

```bash
python src/generate_base.py
```

## 3. Compare decoding strategies

```bash
python src/compare_decoding.py
```

This compares greedy decoding, low temperature, balanced sampling, and high temperature.

## 4. Inspect logits and next-token probabilities

```bash
python src/inspect_logits.py
```

This prints the top candidate tokens for the next generated token.

## 5. Fine-tune with SFT + LoRA

```bash
python src/train_lora.py
```

Output adapter will be saved to:

```text
outputs/lora_adapter/
```

## 6. Evaluate base vs tuned model

```bash
python src/evaluate.py
```

Results are saved to:

```text
outputs/evaluation_results.md
```

## 7. Chat with the tuned adapter

```bash
python src/chat_cli.py
```

Type `exit` to stop.

## Important notes

This is a learning project, not a production model. The dataset is intentionally tiny. Real fine-tuning needs more high-quality data, proper evaluation, reproducibility tracking, safety checks, and licensing review.

## Concepts demonstrated

- Tokenization
- Chat templates
- Causal language modeling
- Greedy decoding
- Temperature
- Top-p sampling
- Logits and softmax probabilities
- Instruction-response data
- SFT
- LoRA
- Base vs tuned model evaluation
- Adapter saving and reloading
