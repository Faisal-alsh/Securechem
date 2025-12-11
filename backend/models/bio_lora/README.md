# Biology LoRA Adapter

This directory should contain the trained LoRA adapter for the biology research assistant.

## Required Files

If you have a trained LoRA adapter, place the following files here:
- `adapter_config.json`
- `adapter_model.bin` or `adapter_model.safetensors`

## Training Your Own Adapter

To train a biology-specific LoRA adapter:

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

# Create PEFT model
model = get_peft_model(base_model, lora_config)

# Train on biology data...
# Save adapter
model.save_pretrained("./bio_lora")
```

## Note

If no adapter is present, the system will use the base model without specialization.
