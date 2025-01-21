# Capítulo 9 - Implementação Básica de RAG

## Introdução

No capítulo anterior, exploramos os fundamentos teóricos do RAG (Retrieval-Augmented Generation). Agora, vamos colocar a mão na massa e implementar um sistema RAG completo do zero. Vamos usar como exemplo prático um assistente virtual para programação que pode responder perguntas baseadas em documentos técnicos, relatórios e histórico de manejo.

## Preparando o Ambiente

Primeiro, vamos configurar nosso ambiente com todas as dependências necessárias:

```python
# Instalação das bibliotecas necessárias
!pip install langchain chromadb python-dotenv openai tiktoken faiss-cpu

# Imports fundamentais
import os
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv

# Carregando variáveis de ambiente
load_dotenv()
```

## Estrutura do Projeto

Vamos organizar nosso projeto de forma modular e escalável:

```
projeto_rag/
├── data/
│   ├── documentos/
│   └── indices/
├── src/
│   ├── __init__.py
│   ├── loader.py
│   ├── processor.py
│   ├── indexer.py
│   └── rag.py
├── .env
└── main.py
```

## 1. Carregamento de Documentos

Vamos implementar um carregador flexível que suporte diferentes tipos de documentos:

```python
# src/loader.py
class DocumentLoader:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        
    def load_documents(self):
        """Carrega documentos de diferentes formatos"""
        loaders = {
            '.txt': DirectoryLoader(
                self.directory_path,
                glob="**/*.txt",
                loader_cls=TextLoader
            ),
            '.pdf': DirectoryLoader(
                self.directory_path,
                glob="**/*.pdf",
                loader_cls=PDFLoader
            ),
            '.csv': CSVLoader(self.directory_path)
        }
        
        documents = []
        for loader in loaders.values():
            try:
                documents.extend(loader.load())
            except Exception as e:
                print(f"Erro ao carregar documentos: {e}")
                
        return documents
```

## 2. Processamento de Texto

O processamento adequado dos documentos é crucial para a eficácia do RAG:

```python
# src/processor.py
class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " "]
        )
    
    def process_documents(self, documents):
        """Processa e divide documentos em chunks"""
        chunks = self.text_splitter.split_documents(documents)
        
        # Adiciona metadados úteis
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                'chunk_id': i,
                'source_type': 'technical_doc',
                'processado_em': datetime.now().isoformat()
            })
        
        return chunks
```

## 3. Indexação e Armazenamento

A indexação eficiente é fundamental para recuperação rápida:

```python
# src/indexer.py
class DocumentIndexer:
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        
    def create_index(self, chunks, persist_directory="./data/indices"):
        """Cria e persiste índice vetorial"""
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embedding_model
        )
        
        # Persistir índice
        vectorstore.save_local(persist_directory)
        return vectorstore
    
    def load_index(self, persist_directory="./data/indices"):
        """Carrega índice existente"""
        if os.path.exists(persist_directory):
            return FAISS.load_local(
                persist_directory,
                self.embedding_model
            )
        raise FileNotFoundError("Índice não encontrado")
```

## 4. Implementação do RAG

Agora vamos juntar tudo em uma classe RAG completa:

```python
# src/rag.py
class RAGSystem:
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0):
        self.llm = OpenAI(
            model_name=model_name,
            temperature=temperature
        )
        self.retriever = None
        self.qa_chain = None
        
    def setup_retriever(self, vectorstore, search_type="mmr", **kwargs):
        """Configura o retriever com estratégia de busca"""
        search_kwargs = {
            "k": kwargs.get("k", 4),
            "fetch_k": kwargs.get("fetch_k", 20),
            "lambda_mult": kwargs.get("lambda_mult", 0.5)
        }
        
        self.retriever = vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
    def setup_qa_chain(self):
        """Configura a chain de pergunta e resposta"""
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            return_source_documents=True
        )
        
    def query(self, question):
        """Processa uma pergunta e retorna resposta com fontes"""
        if not self.qa_chain:
            raise ValueError("QA Chain não configurada")
            
        result = self.qa_chain(question)
        
        return {
            "resposta": result["result"],
            "documentos_fonte": [
                doc.metadata for doc in result["source_documents"]
            ]
        }
```

## 5. Uso Prático

Vamos ver como usar nosso sistema RAG:

```python
# main.py
def main():
    # Inicialização
    loader = DocumentLoader("./data/documentos")
    processor = DocumentProcessor()
    indexer = DocumentIndexer()
    rag = RAGSystem()
    
    # Pipeline de processamento
    documents = loader.load_documents()
    chunks = processor.process_documents(documents)
    vectorstore = indexer.create_index(chunks)
    
    # Configuração do RAG
    rag.setup_retriever(vectorstore)
    rag.setup_qa_chain()
    
    # Exemplo de uso
    pergunta = """
    Quando criarmos uma classe quais são as melhores práticas do clean code ?
    """
    
    resposta = rag.query(pergunta)
    print(f"Resposta: {resposta['resposta']}")
    print("\nFontes utilizadas:")
    for doc in resposta['documentos_fonte']:
        print(f"- {doc['source']}")

if __name__ == "__main__":
    main()
```

## Considerações de Performance

### Otimização de Memória

Para sistemas com grande volume de documentos:

```python
class OptimizedRAGSystem(RAGSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
        
    def query_with_cache(self, question):
        """Implementa cache de respostas frequentes"""
        cache_key = hash(question)
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        result = self.query(question)
        self.cache[cache_key] = result
        return result
```

### Monitoramento de Recursos

```python
def monitor_resources():
    """Monitora uso de recursos do sistema"""
    import psutil
    
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    return {
        "cpu_uso": cpu_percent,
        "memoria_total": memory.total,
        "memoria_disponivel": memory.available,
        "memoria_uso_percentual": memory.percent
    }
```

## Próximos Passos

No próximo capítulo, vamos explorar técnicas avançadas de RAG, incluindo:
- Reranking de resultados
- Prompts dinâmicos
- Avaliação de qualidade das respostas
- Integração com fontes externas

## Recursos Adicionais

Documentação Oficial LangChain RAG
: https://python.langchain.com/docs/use_cases/question_answering

FAISS Documentation
: https://github.com/facebookresearch/faiss/wiki

OpenAI Embeddings Guide
: https://platform.openai.com/docs/guides/embeddings

LangChain Vectorstores Guide
: https://python.langchain.com/docs/modules/data_connection/vectorstores/
