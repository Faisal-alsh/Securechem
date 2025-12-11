# LoRA Fine-Tuning Guide

This guide explains how to prepare training data and fine-tune domain-specific LoRA adapters for the chemistry and biology research assistants.

## 📁 Training Data Location

Place your training data in these directories:

- **Chemistry**: `backend/training_data/chem_training/`
- **Biology**: `backend/training_data/bio_training/`

## 📝 Data Format

### Recommended: Instruction-Response Pairs (JSONL)

Each line is a JSON object with instruction and response fields:

```jsonl
{"instruction": "What are Grignard reagents?", "response": "Grignard reagents are..."}
{"instruction": "Explain PCR", "response": "PCR is a technique..."}
```

**Why this format?**
- Clear task definition for the model
- High-quality alignment with expected behavior
- Easy to create from Q&A pairs, documentation, or expert knowledge

### Sample Data Included

We've provided sample files to get started:
- `backend/training_data/chem_training/sample_instructions.jsonl` (8 chemistry examples)
- `backend/training_data/bio_training/sample_instructions.jsonl` (8 biology examples)

**Note**: These samples are for demonstration only. For effective fine-tuning, you need **1,000+ high-quality examples**.

## 🎯 Data Collection Strategies

### For Chemistry (`chem_training/`)

1. **Lab Protocols**: Convert standard chemistry procedures into Q&A format
   - "How do I prepare X reagent?" → Detailed procedure
   - "What safety precautions for Y?" → Safety guidelines

2. **Textbook Content**: Extract explanations
   - "Explain reaction mechanism Z" → Mechanistic details
   - "What is the principle of technique T?" → Theoretical background

3. **Research Experience**: Document expert knowledge
   - Common troubleshooting questions
   - Best practices and tips
   - Experimental design considerations

4. **Chemistry Forums**: Curate Q&A from reputable sources
   - StackExchange Chemistry
   - Chemical Forums
   - ResearchGate discussions

### For Biology (`bio_training/`)

1. **Molecular Biology Protocols**: Standard techniques
   - "How do I perform Western blot?" → Step-by-step protocol
   - "Explain cell culture procedure" → Detailed methods

2. **Research Papers**: Extract methodology sections
   - Convert methods into Q&A format
   - Include troubleshooting knowledge

3. **Biological Concepts**: Explanations from textbooks
   - "What is CRISPR?" → Detailed explanation
   - "How does gene expression work?" → Mechanistic details

4. **Lab Experience**: Practical knowledge
   - Common problems and solutions
   - Optimization strategies
   - Equipment usage tips

## 📊 Data Quality Guidelines

### Minimum Requirements

- **Quantity**: At least 1,000 instruction-response pairs
- **Quality**: Accurate, verified information only
- **Coverage**: Diverse topics within the domain
- **Format**: Consistent JSONL structure

### Quality Checklist

✓ **Accurate**: All scientific information is correct
✓ **Complete**: Responses fully answer the question
✓ **Concise**: Clear and to-the-point (not overly verbose)
✓ **Consistent**: Similar style across all examples
✓ **Domain-Specific**: Focus on chemistry OR biology, not mixed
✓ **Practical**: Include actionable procedures when relevant
✓ **Safe**: Include appropriate safety information

### What to Avoid

✗ Mixing chemistry and biology data in one file
✗ Outdated or deprecated techniques
✗ Unverified information from unreliable sources
✗ Overly general or vague responses
✗ Copy-pasted content without verification
✗ Procedures without safety considerations

## 🚀 Training Process

### Step 1: Prepare Your Data

Create a JSONL file with your training examples:

```bash
# For chemistry
nano backend/training_data/chem_training/my_chemistry_data.jsonl
```

Format:
```jsonl
{"instruction": "Your question here", "response": "Your detailed answer here"}
{"instruction": "Another question", "response": "Another answer"}
```

### Step 2: Install Training Dependencies

The training script requires additional packages:

```bash
pip install datasets accelerate
```

### Step 3: Run Training

**Chemistry LoRA:**
```bash
python train_lora.py \
  --domain chemistry \
  --data_file backend/training_data/chem_training/my_chemistry_data.jsonl \
  --epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4
```

**Biology LoRA:**
```bash
python train_lora.py \
  --domain biology \
  --data_file backend/training_data/bio_training/my_biology_data.jsonl \
  --epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4
```

### Training Parameters Explained

- `--domain`: Which domain to train (chemistry or biology)
- `--data_file`: Path to your JSONL training data
- `--epochs`: Number of training passes (3-5 recommended)
- `--batch_size`: Samples per batch (reduce if out of memory)
- `--learning_rate`: How fast the model learns (2e-4 is good default)
- `--lora_r`: LoRA rank, controls adapter size (8 is default)
- `--lora_alpha`: LoRA scaling parameter (32 is default)

### Hardware Requirements

**Minimum (CPU training):**
- 16GB RAM
- Training will be slow (hours to days)

**Recommended (GPU training):**
- NVIDIA GPU with 16GB+ VRAM (RTX 4090, A100, etc.)
- 32GB+ RAM
- Training time: 1-4 hours for 1,000 samples

**Cloud Options:**
- Google Colab Pro (A100 GPU)
- AWS EC2 (g5.xlarge or larger)
- Lambda Labs GPU instances

## 📈 Training Progress

During training, you'll see:

```
Loading base model: mistralai/Mistral-7B-Instruct-v0.2
Model and tokenizer loaded
trainable params: 4,194,304 || all params: 7,245,107,200 || trainable%: 0.0579
Loaded 1000 training examples
Data tokenized
Starting training...
Epoch 1/3: [████████████████████] 100%
...
Training complete. Saving adapter to backend/models/chem_lora
✓ LoRA adapter saved successfully!
```

## ✅ Verify Training Success

After training completes, check that adapter files exist:

```bash
# For chemistry
ls -lh backend/models/chem_lora/
# Should see: adapter_config.json, adapter_model.bin (or .safetensors)

# For biology
ls -lh backend/models/bio_lora/
# Should see: adapter_config.json, adapter_model.bin (or .safetensors)
```

## 🧪 Test Your Adapter

1. **Restart the server** to load the new adapter:
```bash
python run_server.py
```

2. **Send test queries**:

Chemistry test:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "researcher": "Test User",
    "password": "1122",
    "chatting": "What are Grignard reagents?"
  }'
```

Biology test:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "researcher": "Test User",
    "password": "3344",
    "chatting": "Explain PCR"
  }'
```

3. **Evaluate responses**:
   - Are they more domain-specific than base model?
   - Do they follow the training data style?
   - Are they accurate and helpful?

## 🔄 Iterative Improvement

### If Results Are Not Good

1. **Add more data**: 1,000 examples is minimum, 10,000+ is better
2. **Improve data quality**: Review and enhance responses
3. **Increase epochs**: Try 5-10 epochs instead of 3
4. **Adjust learning rate**: Try 1e-4 or 3e-4
5. **Filter data**: Remove low-quality examples

### Best Practices

- **Start small**: Test with 100 examples first
- **Validate continuously**: Check outputs during training
- **Domain focus**: Keep chemistry and biology completely separate
- **Version control**: Save different adapter versions
- **Document**: Keep notes on what data was used for each version

## 📋 Example: Complete Workflow

```bash
# 1. Create chemistry training data
cat > backend/training_data/chem_training/my_data.jsonl << 'EOF'
{"instruction": "What is a Grignard reagent?", "response": "A Grignard reagent..."}
{"instruction": "How do I run column chromatography?", "response": "Column chromatography..."}
EOF

# 2. Add more examples (aim for 1000+)
# Edit the file and add more instruction-response pairs

# 3. Train the adapter
python train_lora.py \
  --domain chemistry \
  --data_file backend/training_data/chem_training/my_data.jsonl \
  --epochs 3

# 4. Verify adapter created
ls backend/models/chem_lora/

# 5. Restart server
python run_server.py

# 6. Test in another terminal
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"researcher": "Test", "password": "1122", "chatting": "What is a Grignard reagent?"}'
```

## 🔐 Security Note

After training, your adapters contain knowledge from your training data but:
- ✓ Each adapter stays isolated (chem vs bio)
- ✓ Access control still enforced via passwords
- ✓ No cross-contamination between domains
- ✓ RAG databases remain separate

## 📚 Additional Resources

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face Training Guide](https://huggingface.co/docs/transformers/training)
- [Fine-tuning Best Practices](https://huggingface.co/blog/fine-tune-llm)

## 🆘 Troubleshooting

### Out of Memory Errors

Reduce batch size:
```bash
python train_lora.py --batch_size 1 ...
```

Or reduce LoRA rank:
```bash
python train_lora.py --lora_r 4 ...
```

### Slow Training on CPU

Use smaller dataset for testing, or use cloud GPU.

### Poor Quality Outputs

- Check training data quality
- Ensure enough diverse examples
- Try more training epochs
- Verify data format is correct

### Adapter Not Loading

- Check file paths match expected locations
- Verify `adapter_config.json` exists
- Restart server after training
- Check server logs for errors

---

**Happy training! 🚀**
