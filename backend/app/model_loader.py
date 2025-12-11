"""
Model Loader Module
Handles loading base model and LoRA adapters for different research domains.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from pathlib import Path
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration for model loading"""

    # System prompt for research assistant
    SYSTEM_PROMPT = """[ROLE: RESEARCH ASSISTANT]
You are a concise and accurate research assistant for expert users.

[INSTRUCTIONS]
- Be concise and accurate.
- Explain approaches and design considerations clearly.
- Provide step-by-step lab or operational procedures.
- If you are uncertain, say so explicitly and indicate what information would resolve the uncertainty.

[CONTEXT]
{context}

[QUERY]
{query}"""

    # Base model configuration
    BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

    # LoRA adapter paths (relative to models directory)
    LORA_ADAPTERS = {
        "chem-expert": "chem_lora",
        "bio-expert": "bio_lora"
    }


class ModelManager:
    """Manages base model and LoRA adapters"""

    def __init__(self, models_base_path: str, use_8bit: bool = True):
        """
        Initialize ModelManager.

        Args:
            models_base_path: Base path for model storage
            use_8bit: Whether to use 8-bit quantization
        """
        self.models_base_path = Path(models_base_path)
        self.use_8bit = use_8bit
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Using device: {self.device}")

        # Initialize tokenizer
        self.tokenizer = None

        # Store loaded models (base + adapters)
        self.base_model = None
        self.loaded_adapters: Dict[str, PeftModel] = {}

        # Configuration
        self.config = ModelConfig()

    def load_tokenizer(self):
        """Load tokenizer for the base model"""
        if self.tokenizer is None:
            logger.info(f"Loading tokenizer: {self.config.BASE_MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.BASE_MODEL_NAME,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_base_model(self):
        """Load the base model with optional quantization"""
        if self.base_model is not None:
            return self.base_model

        logger.info(f"Loading base model: {self.config.BASE_MODEL_NAME}")

        # Configure quantization if using 8-bit
        quantization_config = None
        if self.use_8bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )

        # Load model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config.BASE_MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        self.base_model.eval()
        logger.info("Base model loaded successfully")

        return self.base_model

    def load_adapter(self, adapter_name: str) -> PeftModel:
        """
        Load a LoRA adapter.

        Args:
            adapter_name: Name of the adapter (e.g., 'chem-expert', 'bio-expert')

        Returns:
            PeftModel with adapter loaded
        """
        if adapter_name in self.loaded_adapters:
            logger.info(f"Using cached adapter: {adapter_name}")
            return self.loaded_adapters[adapter_name]

        # Ensure base model is loaded
        base_model = self.load_base_model()

        adapter_path = self.models_base_path / self.config.LORA_ADAPTERS[adapter_name]

        # Check if adapter exists
        if adapter_path.exists():
            logger.info(f"Loading LoRA adapter from: {adapter_path}")
            try:
                model_with_adapter = PeftModel.from_pretrained(
                    base_model,
                    adapter_path,
                    is_trainable=False
                )
                model_with_adapter.eval()
                self.loaded_adapters[adapter_name] = model_with_adapter
                logger.info(f"Adapter {adapter_name} loaded successfully")
                return model_with_adapter
            except Exception as e:
                logger.warning(f"Failed to load adapter {adapter_name}: {e}")
                logger.info("Using base model without adapter")
                return base_model
        else:
            logger.warning(f"Adapter not found at {adapter_path}. Using base model.")
            return base_model

    def generate_response(
        self,
        adapter_name: str,
        query: str,
        context: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate response using specified adapter.

        Args:
            adapter_name: Name of the adapter to use
            query: User query
            context: Retrieved RAG context
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated response text
        """
        # Load tokenizer and model with adapter
        tokenizer = self.load_tokenizer()
        model = self.load_adapter(adapter_name)

        # Format prompt with system instructions, context, and query
        prompt = self.config.SYSTEM_PROMPT.format(
            context=context,
            query=query
        )

        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(model.device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the generated part (remove prompt)
        response = full_response[len(prompt):].strip()

        return response

    def cleanup(self):
        """Clean up loaded models to free memory"""
        logger.info("Cleaning up models...")
        self.loaded_adapters.clear()
        if self.base_model is not None:
            del self.base_model
            self.base_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleanup complete")
