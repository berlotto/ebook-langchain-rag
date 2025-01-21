# Instruções para o projeto


## Versao 1

Este projeto é para construção de um ebook de conteúdo técnico para iniciantes em programação de agentes de inteligencia artificial. 
O tema dele é RAG e LangChain. 
Vai abordar todos os conceitos que são necessários para que um iniciante possa estudar e conseguir construir seus primeiros agentes de forma completa. 
A linguagem do ebook é clara, completa e aprofundada, amigável e com um leve toque de humor, e será uma sequencia de instruções do inicio ao fim onde ao final do ebook o leitor tenha passado por cada conceito para conseguir construir seus agentes. 
Quando for explicar os conceitos nao use tanto código, utilize mais códigos nas partes de exemplos.
Entre os capítulos o conteúdo nao deve ser repetitivo, se um tema ou exemplo já foi abordado em outro capítulo, basta somente citar.
Para mostrar diagramas, utilize código mermaid. 
Para mostrar códigos, utilize Python.
Fale sobre o uso de GPUs também. 
Gere todo conteúdo em Markdown. Cada capítulo será um documento .md separado. 
Analise o documento anexado para verificar se está completo, e adicione ou altere o plano conforme achar necessário para enriquecer o conteúdo.
Ao gerar os capitulos, gere um arquivo com Markdown para cada um, o nome dos arquivos gerados será sempre nesse formato "<numero do capitulo>-<titulo do capitulo>.md" e você apresentará em formato renderizado para eu poder verificar o conteúdo antes de utilizá-lo.
Sempre que falar sobre um novo conceito, explique este conceito: o que é, para que serve, aprofunde pq o usuário nao sabe o que é. Precisamos que ao final do ebook ele conheça e saiba o que é cada conceito importante.
Quando tiver alguma necessidade de criar alguma lista aninhada, utilize alguma outra formatação que nao seja lista aninhada mas tenha a mesma entrega de valor e sentido para o leitor. 
Quando você adicionar 'Recursos Adicionais' ao capítulo, sempre adicione a URL do recurso e utilize Definition Lists para ficar melhor formatado.

## Versão 2

```markdown
# Projeto de Ebook sobre Rag e LangChain

## Objetivo Principal
Criar um ebook técnico completo sobre RAG e LangChain para iniciantes em programação de agentes de IA, garantindo que ao final o leitor tenha domínio dos conceitos e seja capaz de construir seus próprios agentes.

## Estrutura e Formato
- Formato: Arquivos Markdown (.md) separados por capítulo
- Nomenclatura: `<numero>-<titulo>.md` (ex: `1-Introducao-ao-RAG.md`)
- Renderização: Apresentação formatada para revisão antes da utilização

## Diretrizes de Conteúdo

### Abordagem Pedagógica
- Linguagem clara, completa e aprofundada
- Tom amigável com toques sutis de humor
- Progressão lógica dos conceitos
- Foco em construção gradual do conhecimento

### Explicação de Conceitos
- Definição completa: o que é, para que serve, como funciona
- Contextualização prática
- Aprofundamento adequado para iniciantes
- Evitar pressupostos de conhecimento prévio

### Código e Exemplos
- Conceitos: Foco em explicação textual e diagramas
- Exemplos práticos: Uso extensivo de código Python
- Diagramas: Utilizar código Mermaid
- Hardware: Incluir especificações e uso de GPUs
- Utilize exemplos diversos da vida cotidiana de problemas a resolver, variando-os entre capítulos

### Formatação
- Evitar listas aninhadas - usar formatos alternativos equivalentes
- Recursos Adicionais: Usar Definition Lists com URLs completas
- Manter consistência visual entre capítulos

### Controle de Qualidade
- Evitar redundância entre capítulos
- Referenciar conteúdo anterior quando necessário
- Analisar documentação existente para completude
- Sugerir melhorias e adições ao plano original

## Resultado Esperado
Um ebook que capacite completamente iniciantes a:
- Compreender todos os conceitos fundamentais
- Implementar agentes de IA práticos
- Utilizar RAG e LangChain efetivamente
- Desenvolver projetos próprios com confiança
```