# Capítulo 10 - RAG Avançado e Otimizações

## Introdução

Nos capítulos anteriores, exploramos os fundamentos do RAG e sua implementação básica. Agora, vamos mergulhar nas técnicas avançadas que podem transformar um sistema RAG básico em uma solução robusta e altamente eficiente. Este capítulo é especialmente relevante para quem trabalha com grandes volumes de dados e precisa de respostas precisas e contextualizadas.

## Técnicas Avançadas de Recuperação

### Reranking Semântico

O reranking é uma técnica que melhora significativamente a qualidade dos documentos recuperados através de uma segunda fase de análise:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Configurando o reranker
base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
compressor = LLMChainExtractor.from_llm(llm)

# Criando retriever com reranking
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Uso
compressed_docs = compression_retriever.get_relevant_documents(
    "Como calcular a taxa de lotação em uma pastagem?"
)
```

### Hybrid Search

Combina busca por embeddings com busca por palavras-chave:

```python
from langchain.retrievers import HybridRetriever
from langchain.retrievers import BM25Retriever

# Configurando retrievers
embedding_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
bm25_retriever = BM25Retriever.from_documents(documents)

# Criando hybrid retriever
hybrid_retriever = HybridRetriever(
    retrievers=[embedding_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # Peso para cada retriever
)
```

## Otimização de Prompts

### Template Dinâmico com Contexto

```python
from langchain.prompts import PromptTemplate

# Template avançado com instruções específicas
template = """
Contexto: {context}

Pergunta: {question}

Instruções:
1. Analise cuidadosamente o contexto fornecido
2. Identifique informações relevantes para a pergunta
3. Forneça uma resposta estruturada e fundamentada
4. Cite as partes específicas do contexto que suportam sua resposta
5. Se houver informações incompletas ou ambíguas, indique claramente

Resposta:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)
```

### Chunking Inteligente

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def custom_length_function(text: str) -> int:
    """Função personalizada para calcular o tamanho do texto"""
    # Remove espaços em branco extras
    text = re.sub(r'\s+', ' ', text.strip())
    # Conta palavras em vez de caracteres
    return len(text.split())

# Splitter customizado
splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,  # Agora em palavras, não caracteres
    chunk_overlap=20,
    length_function=custom_length_function,
    separators=["\n\n", "\n", ".", "!", "?", ";"]
)
```

## Estratégias de Reranking

### Reranking por Relevância Cruzada

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

def rerank_by_cross_relevance(docs, query, llm):
    """
    Reordena documentos baseado em relevância cruzada
    """
    # Gera variações da query
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm
    )
    
    # Recupera documentos para cada variação
    all_docs = retriever.get_relevant_documents(query)
    
    # Conta frequência de cada documento
    doc_frequencies = {}
    for doc in all_docs:
        doc_id = hash(doc.page_content)
        doc_frequencies[doc_id] = doc_frequencies.get(doc_id, 0) + 1
    
    # Reordena baseado na frequência
    reranked_docs = sorted(
        docs,
        key=lambda x: doc_frequencies.get(hash(x.page_content), 0),
        reverse=True
    )
    
    return reranked_docs
```

## Métricas de Avaliação

### Sistema Completo de Avaliação

```python
from typing import Dict, List
import numpy as np

class RAGEvaluator:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        
    def evaluate_retrieval(self, query: str, relevant_docs: List[str]) -> Dict:
        """
        Avalia a qualidade da recuperação
        """
        retrieved_docs = self.retriever.get_relevant_documents(query)
        
        metrics = {
            "precision": self._calculate_precision(retrieved_docs, relevant_docs),
            "recall": self._calculate_recall(retrieved_docs, relevant_docs),
            "relevance_score": self._calculate_relevance(query, retrieved_docs)
        }
        
        return metrics
    
    def _calculate_precision(self, retrieved, relevant):
        """Calcula precisão da recuperação"""
        relevant_retrieved = set(retrieved) & set(relevant)
        return len(relevant_retrieved) / len(retrieved) if retrieved else 0
    
    def _calculate_recall(self, retrieved, relevant):
        """Calcula recall da recuperação"""
        relevant_retrieved = set(retrieved) & set(relevant)
        return len(relevant_retrieved) / len(relevant) if relevant else 0
    
    def _calculate_relevance(self, query, docs):
        """Calcula score de relevância usando embeddings"""
        query_embedding = self.get_embedding(query)
        doc_embeddings = [self.get_embedding(doc.page_content) for doc in docs]
        
        similarities = [
            np.dot(query_embedding, doc_embedding)
            for doc_embedding in doc_embeddings
        ]
        
        return np.mean(similarities)

# Uso do avaliador
evaluator = RAGEvaluator(llm, retriever)
results = evaluator.evaluate_retrieval(
    query="Como calcular o ganho de peso diário?",
    relevant_docs=known_relevant_docs
)
```

## Otimização de Performance

### Cache Inteligente

```python
from functools import lru_cache
import hashlib

class SmartCache:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.cache = {}
        
    @lru_cache(maxsize=1000)
    def get_documents(self, query: str, **kwargs):
        """
        Recupera documentos com cache inteligente
        """
        # Gera hash da query para usar como chave
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Verifica cache
        if query_hash in self.cache:
            return self.cache[query_hash]
        
        # Recupera documentos
        docs = self.vectorstore.similarity_search(query, **kwargs)
        
        # Armazena no cache
        self.cache[query_hash] = docs
        
        return docs
    
    def clear_cache(self):
        """Limpa o cache"""
        self.cache.clear()
        get_documents.cache_clear()
```

### Processamento Paralelo

```python
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

class ParallelRAG:
    def __init__(self, retrievers: List[Dict]):
        """
        retrievers: Lista de dicts com retriever e peso
        """
        self.retrievers = retrievers
        
    def get_documents(self, query: str, max_workers: int = 3):
        """
        Recupera documentos em paralelo de múltiplos retrievers
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submete tarefas para cada retriever
            future_to_retriever = {
                executor.submit(
                    self._get_weighted_docs, 
                    retriever["retriever"],
                    query,
                    retriever["weight"]
                ): retriever
                for retriever in self.retrievers
            }
            
            # Coleta resultados
            all_docs = []
            for future in future_to_retriever:
                try:
                    docs = future.result()
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"Retriever falhou: {e}")
            
        return self._merge_and_deduplicate(all_docs)
    
    def _merge_and_deduplicate(self, docs):
        """
        Combina e remove duplicatas dos documentos recuperados
        """
        seen = set()
        unique_docs = []
        
        for doc in docs:
            doc_hash = hash(doc.page_content)
            if doc_hash not in seen:
                seen.add(doc_hash)
                unique_docs.append(doc)
        
        return unique_docs
```

## Considerações de Hardware

### GPU e Memória

Para um sistema RAG avançado, considere:

- **GPU**: 
  - Mínimo: RTX 3060 12GB para embeddings locais
  - Recomendado: RTX 4080 16GB ou superior para processamento paralelo
  - Ideal: A100 ou H100 para cargas enterprise

- **RAM**:
  - Mínimo: 32GB para bases médias
  - Recomendado: 64GB para processamento paralelo
  - Ideal: 128GB+ para bases grandes e cache extensivo

- **Armazenamento**:
  - SSD NVMe para índice vetorial
  - Mínimo 500GB para bases médias
  - RAID 0 ou 10 para maior performance

## Próximos Passos

No próximo capítulo, exploraremos questões de Compliance e Ética no uso de sistemas RAG, incluindo privacidade de dados, auditoria e conformidade regulatória.

## Recursos Adicionais

Documentação Oficial de Performance RAG
: https://python.langchain.com/docs/guides/deployments/performance

Guia de Otimização de Retrievers
: https://python.langchain.com/docs/modules/data_connection/retrievers/how_to/custom_retriever

Blog sobre RAG Avançado
: https://www.pinecone.io/learn/advanced-retrieval/

Fórum de Otimização LangChain
: https://github.com/langchain-ai/langchain/discussions/categories/optimization