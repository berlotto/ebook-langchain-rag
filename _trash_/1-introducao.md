# **LLMs (Large Language Models): Conceito e Funcionamento**

**1. O que são LLMs?**  

LLMs (Large Language Models) são modelos de linguagem treinados em grandes volumes de dados textuais para realizar tarefas relacionadas à compreensão e geração de texto. Exemplos incluem GPT (OpenAI), BERT (Google) e LLaMA (Meta). Esses modelos utilizam redes neurais profundas, especialmente arquiteturas baseadas em transformadores, para processar texto e gerar saídas coerentes e contextualmente relevantes.

---

**2. Como Funcionam os LLMs?**

Os LLMs aprendem probabilidades estatísticas e relações entre palavras, frases e conceitos. Eles usam uma arquitetura de **Transformers**, que se baseia em mecanismos de atenção para compreender o contexto das palavras em relação a outras palavras no texto.

1. **Input**: Um texto ou frase é fornecido ao modelo.
2. **Tokenização**: O texto é convertido em uma sequência de tokens, que podem ser palavras, subpalavras ou caracteres, dependendo do modelo.
3. **Embeddings**: Cada token é transformado em um vetor numérico de alta dimensão que representa semanticamente o significado do token.
4. **Processamento**: Os vetores passam por múltiplas camadas de transformadores que refinam o entendimento contextual, utilizando o mecanismo de atenção para identificar quais partes do texto são mais relevantes.
5. **Output**: O modelo retorna previsões, como o próximo token (geração de texto), a classificação de sentimento ou a resposta para uma pergunta.

---

## **3. Conceitos Fundamentais**

### **Tokenização**

- Processo de dividir o texto em unidades menores (tokens) que o modelo pode entender.  
- Exemplo com a palavra **"correram"**:
  - Tokenizador WordPiece: ["correr", "##am"]
  - Tokenizador Byte Pair Encoding (BPE): ["cor", "rer", "am"]

#### **Exemplo prático de tokenização em Python (usando HuggingFace):**

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("Large Language Models are fascinating!")
print(tokens)  # ['large', 'language', 'models', 'are', 'fascinating', '!']
```

### **Embeddings**

- Vetores numéricos que representam semanticamente os tokens.  
- Embeddings traduzem texto em números que o modelo pode manipular.  
- Exemplos de bibliotecas que geram embeddings: Word2Vec, GloVe, e a classe `Embeddings` no LangChain.

#### **Exemplo prático de geração de embeddings:**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("Large Language Models are fascinating!")
print(embedding[:10])  # Vetores numéricos representando a frase
```

### **Fine-tuning vs. Adaptação (Prompt Engineering e In-Context Learning)**

| **Aspecto**         | **Fine-tuning**                                                               | **Adaptação**                                                      |
|----------------------|------------------------------------------------------------------------------|--------------------------------------------------------------------|
| **Definição**        | Ajustar os pesos do modelo usando um conjunto de dados específico.           | Utilizar o modelo sem alterar seus pesos, ajustando apenas os inputs (prompts). |
| **Vantagens**        | Personalização total para tarefas específicas.                              | Mais rápido, menos recursos necessários.                          |
| **Exemplo**          | Treinar GPT-3 para gerar resumos jurídicos.                                 | Usar GPT-3 com prompts bem elaborados para sumarizar textos legais. |

#### **Exemplo de Fine-tuning com HuggingFace:**

```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
training_args = TrainingArguments(output_dir="./results", num_train_epochs=3, per_device_train_batch_size=8)
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset)
trainer.train()
```

#### **Exemplo de Adaptação com Prompt Engineering:**

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased")
result = classifier("I love Large Language Models!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.99}]
```


## **4. Exemplos Práticos**

### **Geração de Texto com GPT-3 (OpenAI API):**

```python
import openai

openai.api_key = "sua-chave-api"

response = openai.Completion.create(
    engine="text-davinci-003",
    prompt="Explique o conceito de LLMs de forma simples.",
    max_tokens=100
)
print(response.choices[0].text.strip())
```

### **Perguntas e Respostas (RAG):**

Combinando embeddings e busca em documentos para responder perguntas com contexto:

```python
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()
docsearch = FAISS.load_local("caminho-do-vetor")
qa = RetrievalQA.from_chain_type(llm=model, retriever=docsearch.as_retriever())

result = qa.run("O que são Large Language Models?")
print(result)
```
