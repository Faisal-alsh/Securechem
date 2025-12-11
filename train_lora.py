#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Chemistry and Biology Research Assistants

This script fine-tunes domain-specific LoRA adapters on chemistry or biology data.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoRATrainer:
    """Handles LoRA adapter fine-tuning"""

    def __init__(
        self,
        domain: str,
        base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        output_dir: str = None,
    ):
        """
        Initialize trainer.

        Args:
            domain: 'chemistry' or 'biology'
            base_model_name: HuggingFace model name
            output_dir: Where to save the adapter
        """
        self.domain = domain
        self.base_model_name = base_model_name

        # Set output directory
        if output_dir is None:
            domain_short = "chem" if domain == "chemistry" else "bio"
            self.output_dir = f"backend/models/{domain_short}_lora"
        else:
            self.output_dir = output_dir

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # System prompt for the domain
        self.system_prompt = """[ROLE: RESEARCH ASSISTANT]
You are a concise and accurate research assistant for expert users.

[INSTRUCTIONS]
- Be concise and accurate.
- Explain approaches and design considerations clearly.
- Provide step-by-step lab or operational procedures.
- If you are uncertain, say so explicitly and indicate what information would resolve the uncertainty."""

        self.tokenizer = None
        self.model = None

    def load_model_and_tokenizer(self):
        """Load base model and tokenizer"""
        logger.info(f"Loading base model: {self.base_model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        logger.info("Model and tokenizer loaded")

    def configure_lora(
        self,
        r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: List[str] = None,
    ):
        """
        Configure LoRA settings.

        Args:
            r: LoRA rank (lower = fewer parameters)
            lora_alpha: LoRA scaling parameter
            lora_dropout: Dropout probability
            target_modules: Which modules to apply LoRA to
        """
        if target_modules is None:
            # Common attention modules for Mistral
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        # Prepare model for training
        if torch.cuda.is_available():
            self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        logger.info("LoRA configuration applied")

    def load_training_data(self, data_file: str) -> Dataset:
        """
        Load training data from JSONL file.

        Expected format:
        {"instruction": "...", "response": "..."}

        Args:
            data_file: Path to JSONL file

        Returns:
            HuggingFace Dataset
        """
        logger.info(f"Loading training data from: {data_file}")

        data = []
        with open(data_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))

        logger.info(f"Loaded {len(data)} training examples")

        # Format data with system prompt
        formatted_data = []
        for item in data:
            # Create conversation format
            conversation = f"{self.system_prompt}\n\n[USER QUERY]\n{item['instruction']}\n\n[ASSISTANT RESPONSE]\n{item['response']}"
            formatted_data.append({"text": conversation})

        return Dataset.from_list(formatted_data)

    def tokenize_data(self, dataset: Dataset, max_length: int = 2048) -> Dataset:
        """
        Tokenize dataset.

        Args:
            dataset: Dataset to tokenize
            max_length: Maximum sequence length

        Returns:
            Tokenized dataset
        """
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )

        logger.info("Data tokenized")
        return tokenized_dataset

    def train(
        self,
        train_dataset: Dataset,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        gradient_accumulation_steps: int = 4,
    ):
        """
        Train the LoRA adapter.

        Args:
            train_dataset: Tokenized training dataset
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            gradient_accumulation_steps: Accumulation steps for larger effective batch
        """
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{self.output_dir}/logs",
            report_to="none",  # Disable wandb/tensorboard
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )

        logger.info("Starting training...")
        trainer.train()

        logger.info(f"Training complete. Saving adapter to {self.output_dir}")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        logger.info("✓ LoRA adapter saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune LoRA adapters for chemistry or biology domains"
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=["chemistry", "biology"],
        help="Domain to train on",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to training data (JSONL format)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for adapter (default: backend/models/{domain}_lora)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = LoRATrainer(
        domain=args.domain,
        base_model_name=args.base_model,
        output_dir=args.output_dir,
    )

    # Load model and tokenizer
    trainer.load_model_and_tokenizer()

    # Configure LoRA
    trainer.configure_lora(r=args.lora_r, lora_alpha=args.lora_alpha)

    # Load and prepare training data
    dataset = trainer.load_training_data(args.data_file)
    tokenized_dataset = trainer.tokenize_data(dataset)

    # Train
    trainer.train(
        train_dataset=tokenized_dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    print("\n" + "="*80)
    print(f"✓ LoRA adapter training complete for {args.domain}!")
    print(f"✓ Adapter saved to: {trainer.output_dir}")
    print("\nTo use this adapter:")
    print(f"  1. Restart the server: python run_server.py")
    print(f"  2. Send requests with the appropriate credentials:")
    if args.domain == "chemistry":
        print('     {"researcher": "...", "password": "1122", "chatting": "..."}')
    else:
        print('     {"researcher": "...", "password": "3344", "chatting": "..."}')
    print("="*80)


if __name__ == "__main__":
    main()
