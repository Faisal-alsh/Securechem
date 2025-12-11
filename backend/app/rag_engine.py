"""
RAG (Retrieval-Augmented Generation) Engine
Implements simple tokenization and text search for context retrieval.
"""

import os
from pathlib import Path
from typing import List, Dict
import re
from collections import Counter
import numpy as np


class RAGDocument:
    """Represents a document chunk in the RAG database"""

    def __init__(self, text: str, metadata: Dict = None):
        self.text = text
        self.metadata = metadata or {}
        self.tokens = self._tokenize(text)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase and split on non-alphanumeric"""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens


class RAGDatabase:
    """Simple RAG database with text search capabilities"""

    def __init__(self, database_path: str):
        self.database_path = Path(database_path)
        self.documents: List[RAGDocument] = []
        self.load_documents()

    def load_documents(self):
        """Load documents from the database directory"""
        if not self.database_path.exists():
            raise FileNotFoundError(f"RAG database not found: {self.database_path}")

        # Load all text files from the database directory
        for file_path in self.database_path.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split content into paragraphs (document chunks)
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

            for idx, paragraph in enumerate(paragraphs):
                metadata = {
                    'source_file': file_path.name,
                    'chunk_id': idx
                }
                doc = RAGDocument(paragraph, metadata)
                self.documents.append(doc)

        print(f"Loaded {len(self.documents)} document chunks from {self.database_path}")

    def _compute_similarity(self, query_tokens: List[str], doc: RAGDocument) -> float:
        """
        Compute similarity between query and document using token overlap.
        Uses TF-IDF-like scoring.
        """
        if not query_tokens or not doc.tokens:
            return 0.0

        # Count query tokens
        query_counter = Counter(query_tokens)
        doc_counter = Counter(doc.tokens)

        # Compute term frequency overlap
        score = 0.0
        for token, query_count in query_counter.items():
            if token in doc_counter:
                # Simple scoring: more matches = higher score
                score += min(query_count, doc_counter[token])

        # Normalize by document length to favor relevant shorter chunks
        if len(doc.tokens) > 0:
            score = score / np.sqrt(len(doc.tokens))

        return score

    def search(self, query: str, top_k: int = 3) -> List[RAGDocument]:
        """
        Search for relevant documents using token-based similarity.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of most relevant documents
        """
        # Tokenize query
        query_tokens = re.findall(r'\b\w+\b', query.lower())

        if not query_tokens:
            return []

        # Compute similarity scores
        doc_scores = []
        for doc in self.documents:
            score = self._compute_similarity(query_tokens, doc)
            doc_scores.append((score, doc))

        # Sort by score (descending) and return top k
        doc_scores.sort(reverse=True, key=lambda x: x[0])
        top_docs = [doc for score, doc in doc_scores[:top_k] if score > 0]

        return top_docs

    def get_context(self, query: str, max_context_length: int = 1000) -> str:
        """
        Retrieve relevant context for a query.

        Args:
            query: Search query
            max_context_length: Maximum context length in characters

        Returns:
            Formatted context string
        """
        relevant_docs = self.search(query, top_k=3)

        if not relevant_docs:
            return "No relevant context found in the knowledge base."

        # Combine document texts
        context_parts = []
        current_length = 0

        for doc in relevant_docs:
            doc_text = doc.text
            if current_length + len(doc_text) > max_context_length:
                # Truncate if needed
                remaining = max_context_length - current_length
                if remaining > 100:  # Only add if meaningful amount remains
                    doc_text = doc_text[:remaining] + "..."
                    context_parts.append(doc_text)
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        context = "\n\n".join(context_parts)
        return context


class RAGManager:
    """Manages multiple RAG databases for different domains"""

    def __init__(self, base_data_path: str):
        self.base_data_path = Path(base_data_path)
        self.databases: Dict[str, RAGDatabase] = {}

    def load_database(self, database_name: str) -> RAGDatabase:
        """
        Load a RAG database by name.

        Args:
            database_name: Name of the database (e.g., 'chem_rag', 'bio_rag')

        Returns:
            RAGDatabase instance
        """
        if database_name in self.databases:
            return self.databases[database_name]

        database_path = self.base_data_path / database_name
        rag_db = RAGDatabase(database_path)
        self.databases[database_name] = rag_db

        return rag_db

    def get_context(self, database_name: str, query: str) -> str:
        """
        Get context from a specific database.

        Args:
            database_name: Name of the database
            query: Search query

        Returns:
            Retrieved context
        """
        rag_db = self.load_database(database_name)
        context = rag_db.get_context(query)
        return context
