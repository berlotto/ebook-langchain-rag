"""
RAG com Fontes Dinâmicas em Tempo Real usando Ollama
Uma implementação de sistema RAG que permite atualização dinâmica de fontes de conhecimento.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib
import asyncio
import aiohttp
import json
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.callbacks import get_openai_callback
from langchain.schema import Document
from pydantic import BaseModel, Field

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Source(BaseModel):
    """Modelo para fonte de dados dinâmica"""
    name: str
    type: str = Field(..., description="Tipo da fonte: 'rss', 'api', 'file', etc")
    url: Optional[str] = None
    update_interval: int = Field(default=300, description="Intervalo de atualização em segundos")
    last_update: Optional[datetime] = None
    credentials: Optional[Dict[str, str]] = None
    parameters: Optional[Dict[str, Any]] = None

class DocumentProcessor:
    """Processa documentos para indexação"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?"]
        )
        
    def process_text(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """Processa texto em chunks com metadados"""
        try:
            chunks = self.text_splitter.create_documents(
                texts=[text],
                metadatas=[metadata]
            )
            return chunks
        except Exception as e:
            logger.error(f"Erro ao processar texto: {str(e)}")
            return []

class SourceManager:
    """Gerencia fontes de dados dinâmicas"""
    
    def __init__(self):
        self.sources: Dict[str, Source] = {}
        self.processor = DocumentProcessor()
        
    def add_source(self, source: Source):
        """Adiciona nova fonte"""
        self.sources[source.name] = source
        logger.info(f"Fonte adicionada: {source.name}")
        
    def remove_source(self, source_name: str):
        """Remove fonte existente"""
        if source_name in self.sources:
            del self.sources[source_name]
            logger.info(f"Fonte removida: {source_name}")
            
    async def fetch_source_data(self, source: Source) -> List[Document]:
        """Busca dados de uma fonte"""
        try:
            if source.type == "rss":
                return await self._fetch_rss(source)
            elif source.type == "api":
                return await self._fetch_api(source)
            elif source.type == "file":
                return await self._fetch_file(source)
            else:
                logger.warning(f"Tipo de fonte não suportado: {source.type}")
                return []
        except Exception as e:
            logger.error(f"Erro ao buscar dados da fonte {source.name}: {str(e)}")
            return []
            
    async def _fetch_rss(self, source: Source) -> List[Document]:
        """Busca dados de feed RSS"""
        async with aiohttp.ClientSession() as session:
            async with session.get(source.url) as response:
                if response.status == 200:
                    import feedparser
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    documents = []
                    for entry in feed.entries:
                        metadata = {
                            "source": source.name,
                            "type": "rss",
                            "title": entry.get("title", ""),
                            "published": entry.get("published", ""),
                            "url": entry.get("link", "")
                        }
                        
                        text = f"{entry.get('title', '')}\n{entry.get('description', '')}"
                        chunks = self.processor.process_text(text, metadata)
                        documents.extend(chunks)
                    
                    return documents
                else:
                    logger.error(f"Erro ao buscar RSS: {response.status}")
                    return []
                    
    async def _fetch_api(self, source: Source) -> List[Document]:
        """Busca dados de API REST"""
        headers = source.credentials if source.credentials else {}
        params = source.parameters if source.parameters else {}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                source.url,
                headers=headers,
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    documents = []
                    if isinstance(data, list):
                        for item in data:
                            metadata = {
                                "source": source.name,
                                "type": "api",
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            text = json.dumps(item, ensure_ascii=False)
                            chunks = self.processor.process_text(text, metadata)
                            documents.extend(chunks)
                    else:
                        metadata = {
                            "source": source.name,
                            "type": "api",
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        text = json.dumps(data, ensure_ascii=False)
                        chunks = self.processor.process_text(text, metadata)
                        documents.extend(chunks)
                    
                    return documents
                else:
                    logger.error(f"Erro ao buscar API: {response.status}")
                    return []
                    
    async def _fetch_file(self, source: Source) -> List[Document]:
        """Busca dados de arquivo local"""
        try:
            path = Path(source.url)
            if not path.exists():
                logger.error(f"Arquivo não encontrado: {source.url}")
                return []
                
            if path.suffix == '.csv':
                df = pd.read_csv(path)
                documents = []
                
                for _, row in df.iterrows():
                    metadata = {
                        "source": source.name,
                        "type": "file",
                        "filename": path.name,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    text = row.to_json(force_ascii=False)
                    chunks = self.processor.process_text(text, metadata)
                    documents.extend(chunks)
                    
                return documents
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    
                metadata = {
                    "source": source.name,
                    "type": "file",
                    "filename": path.name,
                    "timestamp": datetime.now().isoformat()
                }
                
                return self.processor.process_text(text, metadata)
                
        except Exception as e:
            logger.error(f"Erro ao ler arquivo {source.url}: {str(e)}")
            return []

class DynamicVectorStore:
    """Gerencia armazenamento e atualização de vetores"""
    
    def __init__(self):
        # Usando modelo BERT multilíngue para embeddings
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/multilingual-MiniLM-L12-v2"
        )
        self.vector_store = None
        self.document_hashes = set()
        
    def initialize_store(self):
        """Inicializa store vazio"""
        self.vector_store = FAISS(
            embedding_function=self.embedding_model,
            index=None
        )
        logger.info("Vector store inicializado")
        
    def _generate_hash(self, document: Document) -> str:
        """Gera hash único para documento"""
        content = f"{document.page_content}{json.dumps(document.metadata)}"
        return hashlib.md5(content.encode()).hexdigest()
        
    def update_documents(self, documents: List[Document]) -> int:
        """Atualiza documentos no store, retorna número de novos docs"""
        if not self.vector_store:
            self.initialize_store()
            
        new_documents = []
        for doc in documents:
            doc_hash = self._generate_hash(doc)
            if doc_hash not in self.document_hashes:
                new_documents.append(doc)
                self.document_hashes.add(doc_hash)
                
        if new_documents:
            self.vector_store.add_documents(new_documents)
            logger.info(f"Adicionados {len(new_documents)} novos documentos")
                
        return len(new_documents)
        
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Realiza busca por similaridade com filtros opcionais"""
        if not self.vector_store:
            logger.warning("Vector store não inicializado")
            return []
            
        try:
            docs = self.vector_store.similarity_search(
                query,
                k=k,
                filter=filter_metadata
            )
            return docs
        except Exception as e:
            logger.error(f"Erro na busca: {str(e)}")
            return []

class DynamicRAG:
    """Sistema RAG com atualização dinâmica de fontes"""
    
    def __init__(
        self,
        llm_model: Optional[str] = "llama2",
        update_interval: int = 300
    ):
        self.source_manager = SourceManager()
        self.vector_store = DynamicVectorStore()
        # Usando Llama2 através do Ollama
        self.llm = Ollama(model=llm_model)
        self.update_interval = update_interval
        self._update_task = None
        
    def add_source(self, source: Source):
        """Adiciona nova fonte de dados"""
        self.source_manager.add_source(source)
        
    def remove_source(self, source_name: str):
        """Remove fonte de dados"""
        self.source_manager.remove_source(source_name)
        
    async def _update_sources(self):
        """Atualiza todas as fontes periodicamente"""
        while True:
            try:
                all_documents = []
                for source in self.source_manager.sources.values():
                    if (not source.last_update or
                        (datetime.now() - source.last_update).seconds >= source.update_interval):
                        
                        documents = await self.source_manager.fetch_source_data(source)
                        all_documents.extend(documents)
                        source.last_update = datetime.now()
                
                if all_documents:
                    new_docs = self.vector_store.update_documents(all_documents)
                    logger.info(f"Adicionados {new_docs} novos documentos")
                    
            except Exception as e:
                logger.error(f"Erro na atualização: {str(e)}")
                
            await asyncio.sleep(self.update_interval)
            
    def start_updates(self):
        """Inicia processo de atualização automática"""
        if not self._update_task:
            self._update_task = asyncio.create_task(self._update_sources())
            logger.info("Iniciada atualização automática")
            
    def stop_updates(self):
        """Para processo de atualização automática"""
        if self._update_task:
            self._update_task.cancel()
            self._update_task = None
            logger.info("Atualização automática interrompida")
            
    async def query(
        self,
        question: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Processa uma pergunta e retorna resposta com contexto"""
        try:
            # Recupera documentos relevantes
            relevant_docs = self.vector_store.similarity_search(
                question,
                k=4,
                filter_metadata=filter_metadata
            )
            
            if not relevant_docs:
                return {
                    "answer": "Desculpe, não encontrei informações relevantes para sua pergunta.",
                    "sources": [],
                    "context": []
                }
                
            # Prepara contexto
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Prepara prompt
            prompt = f"""
            Responda a seguinte pergunta usando apenas o contexto fornecido.
            Se não puder responder apenas com o contexto, diga que não tem informações suficientes.
            
            Contexto:
            {context}
            
            Pergunta: {question}
            
            Resposta:
            """
            
            # Gera resposta usando Llama2
            response = self.llm.predict(prompt)
            
            return {
                "answer": response.strip(),
                "sources": [doc.metadata for doc in relevant_docs],
                "context": [doc.page_content for doc in relevant_docs]
            }
            
        except Exception as e:
            logger.error(f"Erro ao processar query: {str(e)}")
            return {
                "error": f"Erro ao processar sua pergunta: {str(e)}",
                "sources": [],
                "context": []
            }

# Exemplo de uso
async def main():
    # Inicializa sistema RAG
    rag = DynamicRAG()
    
    # Configura fontes
    sources = [
        Source(
            name="tech_news",
            type="rss",
            url="https://exemplo.com/feed.xml",
            update_interval=600
        ),
        Source(
            name="docs_tecnicos",
            type="file",
            url="./dados/documentos_tecnicos.pdf",
            update_interval=3600
        )
    ]
    
    # Adiciona fontes
    for source in sources:
        rag.add_source(source)
    
    # Inicia atualizações automáticas
    rag.start_updates()
    
    # Exemplo de consulta
    result = await rag.query(
        "Quais são as últimas novidades em desenvolvimento de software?",
        filter_metadata={"type": "rss"}
    )
    
    print("Resposta:", result["answer"])
    print("\nFontes utilizadas:")
    for source in result["sources"]:
        print(f"- {source['title']} ({source['url']})")
    
    # Aguarda algumas atualizações
    await asyncio.sleep(60)
    
    # Para o sistema
    rag.stop_updates()

if __name__ == "__main__":
    # Roda o exemplo
    asyncio.run(main())