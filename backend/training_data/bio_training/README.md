# Biology Training Data

Place your biology-specific training data here for LoRA fine-tuning.

## Recommended Format

### Option 1: Instruction-Response Pairs (JSONL)

Create a file `biology_instructions.jsonl` with entries like:

```jsonl
{"instruction": "What is CRISPR-Cas9?", "response": "CRISPR-Cas9 is a gene editing technology that uses RNA-guided nuclease to make precise cuts in DNA. The guide RNA directs the Cas9 enzyme to a specific genomic location where it creates a double-strand break, enabling targeted modifications through cellular DNA repair mechanisms."}
{"instruction": "Explain PCR", "response": "Polymerase Chain Reaction (PCR) is a technique for amplifying specific DNA sequences. It uses thermal cycling with DNA polymerase, primers, and nucleotides to exponentially replicate target DNA through repeated denaturation, annealing, and extension steps."}
```

### Option 2: Conversational Format (JSONL)

```jsonl
{"messages": [{"role": "user", "content": "How does mitosis work?"}, {"role": "assistant", "content": "Mitosis is cell division that produces two identical daughter cells..."}]}
```

### Option 3: Plain Text Dataset

Create `biology_corpus.txt` with domain-specific text:

```
Cell biology encompasses the study of cellular structure and function.
The cell membrane is a phospholipid bilayer...

Molecular biology techniques include cloning, PCR, and sequencing.
DNA replication is a semi-conservative process...
```

## Training Data Sources

- Research papers in biology
- Molecular biology protocols
- Biology textbook excerpts
- Q&A pairs from biology forums
- Lab techniques and experimental procedures

## Size Recommendations

- **Minimum**: 1,000 instruction-response pairs or 1MB text
- **Recommended**: 10,000+ pairs or 10MB+ text
- **Optimal**: 50,000+ pairs or 100MB+ text

## Data Quality Guidelines

1. **Domain-Specific**: Focus on biology topics
2. **Accurate**: Verify all biological information
3. **Diverse**: Cover various biology subfields (molecular, cellular, genetics, etc.)
4. **Consistent Format**: Use same format throughout
5. **Clean**: Remove formatting artifacts, fix typos

## Example Files

See the training script for how to use your data:
```bash
python train_lora.py --domain biology --data_file training_data/bio_training/biology_instructions.jsonl
```
