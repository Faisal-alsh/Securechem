# Chemistry Training Data

Place your chemistry-specific training data here for LoRA fine-tuning.

## Recommended Format

### Option 1: Instruction-Response Pairs (JSONL)

Create a file `chemistry_instructions.jsonl` with entries like:

```jsonl
{"instruction": "What are Grignard reagents?", "response": "Grignard reagents (RMgX) are organometallic compounds containing a carbon-metal bond between an alkyl or aryl group and magnesium halide. They are strong nucleophiles and bases used in organic synthesis."}
{"instruction": "How do I perform a nucleophilic substitution?", "response": "For SN2 reactions, use a strong nucleophile with a primary alkyl halide in a polar aprotic solvent. The nucleophile attacks from the backside, leading to inversion of configuration. For SN1, use tertiary substrates in polar protic solvents where carbocation formation is favored."}
```

### Option 2: Conversational Format (JSONL)

```jsonl
{"messages": [{"role": "user", "content": "Explain chromatography"}, {"role": "assistant", "content": "Chromatography is a separation technique..."}]}
```

### Option 3: Plain Text Dataset

Create `chemistry_corpus.txt` with domain-specific text:

```
Organic synthesis requires understanding reaction mechanisms.
The SN2 mechanism proceeds via a single concerted step...

Spectroscopy techniques include NMR, IR, and mass spectrometry.
NMR spectroscopy provides structural information...
```

## Training Data Sources

- Research papers in chemistry
- Lab protocols and procedures
- Chemistry textbook excerpts
- Q&A pairs from chemistry forums
- Experimental procedures and safety guidelines

## Size Recommendations

- **Minimum**: 1,000 instruction-response pairs or 1MB text
- **Recommended**: 10,000+ pairs or 10MB+ text
- **Optimal**: 50,000+ pairs or 100MB+ text

## Data Quality Guidelines

1. **Domain-Specific**: Focus on chemistry topics
2. **Accurate**: Verify all chemical information
3. **Diverse**: Cover various chemistry subfields
4. **Consistent Format**: Use same format throughout
5. **Clean**: Remove formatting artifacts, fix typos

## Example Files

See the training script for how to use your data:
```bash
python train_lora.py --domain chemistry --data_file training_data/chem_training/chemistry_instructions.jsonl
```
