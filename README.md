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
│                     FastAPI Backend                         │
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
│  │   Base: Mistral-7B-Instruct-v0.2           │          │
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
```

## 🔑 Access Credentials

| Role | Password | Model | RAG Database |
|------|----------|-------|--------------|
| CHEM_RESEARCHER | `1122` | chem-expert | chemistry_knowledge.txt |
| BIO_RESEARCHER | `3344` | bio-expert | biology_knowledge.txt |

## 📋 API Specification

### POST /chat

**Request:**
```json
{
  "researcher": "Dr. Smith",
  "password": "1122",
  "chatting": "What are Grignard reagents?"
}
```

**Response:**
```json
{
  "answer": "Grignard reagents (RMgX) are organometallic compounds..."
}
```

**Authentication Errors:**
- `401 Unauthorized`: Invalid credentials
- `403 Forbidden`: Role-based access denied

### GET /health

Check server status.

### GET /info

Get information about available models and roles.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd Securechem

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration (Optional)

Copy `.env.example` to `.env` and customize settings:

```bash
cp .env.example .env
```

### 3. Run the Server

```bash
# Start the FastAPI server
python run_server.py
```

The server will start at `http://localhost:8000`

### 4. Test the API

```bash
# Run the test client
python test_client.py
```

Or use curl:

```bash
# Chemistry query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "researcher": "Dr. Smith",
    "password": "1122",
    "chatting": "What are Grignard reagents and how do I prepare them?"
  }'

# Biology query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "researcher": "Dr. Johnson",
    "password": "3344",
    "chatting": "Explain CRISPR-Cas9 gene editing"
  }'
```

## 📁 Project Structure

```
Securechem/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application
│   │   ├── auth.py              # Authentication & authorization
│   │   ├── model_loader.py      # Model & LoRA adapter loading
│   │   └── rag_engine.py        # RAG retrieval engine
│   ├── config/
│   │   └── settings.py          # Configuration management
│   ├── data/
│   │   ├── chem_rag/
│   │   │   └── chemistry_knowledge.txt
│   │   └── bio_rag/
│   │       └── biology_knowledge.txt
│   └── models/
│       ├── chem_lora/           # Chemistry LoRA adapter (if trained)
│       └── bio_lora/            # Biology LoRA adapter (if trained)
├── requirements.txt
├── run_server.py                # Server startup script
├── test_client.py               # Test client
├── .env.example                 # Environment variables template
└── README.md
```

## 🧪 System Prompt

The LLM uses this system prompt for all responses:

```
[ROLE: RESEARCH ASSISTANT]
You are a concise and accurate research assistant for expert users.

[INSTRUCTIONS]
- Be concise and accurate.
- Explain approaches and design considerations clearly.
- Provide step-by-step lab or operational procedures.
- If you are uncertain, say so explicitly and indicate what
  information would resolve the uncertainty.
```

## 🔧 Technical Stack

- **Python**: 3.10+
- **FastAPI**: Modern async web framework
- **Uvicorn**: ASGI server
- **Transformers**: HuggingFace transformers library
- **PEFT**: Parameter-Efficient Fine-Tuning (LoRA)
- **Accelerate**: Model acceleration
- **PyTorch**: Deep learning framework

## 🎯 RAG Implementation

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

## 🐛 Troubleshooting

### Model Loading Issues

If you get CUDA/memory errors:
- Set `USE_8BIT_QUANTIZATION=False` in `.env` (CPU mode)
- Reduce `MAX_NEW_TOKENS` to lower memory usage
- Ensure you have sufficient RAM (8GB+ recommended)

### Port Already in Use

Change the port in `run_server.py` or `.env`:
```python
API_PORT=8001
```

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## 📝 API Testing with Postman/Insomnia

**Endpoint**: `POST http://localhost:8000/chat`

**Headers**:
```
Content-Type: application/json
```

**Body** (Chemistry):
```json
{
  "researcher": "Dr. Smith",
  "password": "1122",
  "chatting": "Explain SN1 and SN2 reactions"
}
```

**Body** (Biology):
```json
{
  "researcher": "Dr. Johnson",
  "password": "3344",
  "chatting": "What is PCR and how does it work?"
}
```

## 📄 License

This project is provided as-is for research and educational purposes.

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows existing style
- Security features are maintained
- Domain isolation is preserved
- Tests pass

## ⚠️ Security Notes

- **Production Use**: Replace simple password authentication with proper auth (JWT, OAuth2)
- **Environment Variables**: Never commit `.env` files with real credentials
- **HTTPS**: Use HTTPS in production
- **Rate Limiting**: Add rate limiting for production deployment
- **Input Validation**: Current implementation has basic validation; enhance for production

## 📚 References

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Mistral AI Models](https://huggingface.co/mistralai)

---

**Built with ❤️ for secure research collaboration**