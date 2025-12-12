# Secure Research Assistant Backend

A secure backend system for two isolated research-assistant LLMs: **chem-expert** (chemistry) and **bio-expert** (biology). Each model uses a shared base model (Mistral-7B-Instruct) with domain-specific LoRA adapters and RAG databases.

## 🔒 Security Features

- **Strict Access Control**: Password-based authentication with role enforcement
- **Domain Isolation**: Each model has separate LoRA adapter and RAG database
- **No Cross-Access**: Chemistry and biology data are completely isolated
- **Secure Routing**: Requests are automatically routed to correct domain based on credentials

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐      ┌──────────────────┐           │
│  │  Authentication  │      │   RAG Manager    │           │
│  │   & Authorization│      │   (Retrieval)    │           │
│  └──────────────────┘      └──────────────────┘           │
│           │                         │                       │
│           ▼                         ▼                       │
│  ┌─────────────────────────────────────────────┐          │
│  │         Model Manager (PEFT)                │          │
│  │   Base: gpt2                                           │          │
│  │                                              │          │
│  │  ┌──────────────┐    ┌──────────────┐     │          │
│  │  │  Chem LoRA   │    │  Bio LoRA    │     │          │
│  │  │  Adapter     │    │  Adapter     │     │          │
│  │  └──────────────┘    └──────────────┘     │          │
│  └─────────────────────────────────────────────┘          │
│           │                         │                       │
│           ▼                         ▼                       │
│  ┌──────────────┐         ┌──────────────┐                │
│  │  Chem RAG DB │         │  Bio RAG DB  │                │
│  │  (Chemistry  │         │  (Biology    │                │
│  │   Context)   │         │   Context)   │                │
│  └──────────────┘         └──────────────┘                │
└─────────────────────────────────────────────────────────────┘




- **Python**: 3.10+
- **FastAPI**: Modern async web framework
- **Uvicorn**: ASGI server
- **Transformers**: HuggingFace transformers library
- **PEFT**: Parameter-Efficient Fine-Tuning (LoRA)
- **Accelerate**: Model acceleration
- **PyTorch**: Deep learning framework

## 🎯 RAG Implementation
 - FAISS VDB

The system uses **token-based text search** for RAG retrieval:

1. **Document Loading**: Text files are split into paragraph chunks
2. **Tokenization**: Simple regex-based tokenization (lowercase, alphanumeric)
3. **Similarity Scoring**: TF-IDF-like scoring based on token overlap
4. **Context Retrieval**: Top-3 most relevant chunks are concatenated
5. **Prompt Construction**: Context + Query + System Prompt

## 🔐 Security Implementation

### Access Control Flow

1. **Authentication**: Validate password and map to role
2. **Authorization**: Ensure role has access to requested resource
3. **Isolation**: Each role can only access its designated RAG database
4. **Model Routing**: Automatically load correct LoRA adapter based on role

### Domain Isolation

- Chemistry researchers (`password: 1122`) → `chem-expert` → `chem_rag`
- Biology researchers (`password: 3344`) → `bio-expert` → `bio_rag`

No mechanism exists for cross-domain data access or adapter sharing.

## 🚀 Adding LoRA Adapters

By default, the system works with the base model. To add trained LoRA adapters:

1. **Train your adapter** (see `backend/models/*/README.md` for examples)
2. **Place adapter files** in the appropriate directory:
   - Chemistry: `backend/models/chem_lora/`
   - Biology: `backend/models/bio_lora/`
3. **Required files**:
   - `adapter_config.json`
   - `adapter_model.bin` or `adapter_model.safetensors`
4. **Restart server** to load adapters

The system will automatically detect and load adapters if present.

## 📊 Extending RAG Databases

To add more knowledge to the RAG databases:

1. **Create/edit text files** in:
   - Chemistry: `backend/data/chem_rag/`
   - Biology: `backend/data/bio_rag/`
2. **Format**: Plain text, paragraphs separated by blank lines
3. **Restart server** to reload documents

