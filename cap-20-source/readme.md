# RAG Dinâmico em Tempo Real

Este é o projeto prático do Capítulo 20 do ebook "Dominando RAG e LangChain: Do Básico ao Avançado". 

## Sobre o Projeto

Um sistema RAG (Retrieval-Augmented Generation) que:
- Atualiza sua base de conhecimento em tempo real
- Monitora múltiplas fontes de dados simultaneamente
- Roda 100% local usando Ollama/Llama2
- Implementa deduplicação inteligente
- Usa processamento assíncrono para máxima performance

## Requisitos

- Python 3.9+
- Ollama instalado com modelo Llama2
- Dependências Python listadas em requirements.txt

## Instalação

1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/dynamic-rag.git
cd dynamic-rag
```

2. Instale as dependências
```bash
pip install -r requirements.txt
```

3. Instale e configure o Ollama
```bash
curl https://ollama.ai/install.sh | sh
ollama pull llama2
```

## Uso

Execute o script principal:
```bash
python dynamic_rag.py
```

## Quer Aprender Mais?

Este é apenas um dos muitos projetos práticos do ebook "Dominando RAG e LangChain: Do Básico ao Avançado". 

Para realmente dominar estas tecnologias, [acesse aqui](URL_DO_EBOOK) e aprenda:
- Fundamentos de LLMs
- LangChain do básico ao avançado
- RAG e sistemas de busca semântica
- Embeddings e vetorização
- Integração com HuggingFace e Ollama
- E muito mais!

## Licença

MIT

---
*Este projeto é parte do material educacional do ebook "Dominando RAG e LangChain: Do Básico ao Avançado"*