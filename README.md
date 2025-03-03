# LLM Agents and Tools
 ---
 Tags:  #literature 
 Description:  Curso do Pycode BR
 Theme: [[Mestrado ITA 👨🏽‍🏫]],  [[Machine Learning]], [[Integrando com API GPT(OpenAI)]]
## ID: 20250301233543
---


![Canvas](arq_LLM.png)


# <font color="#c3d69b">Descrição breve: </font>


Dois conceitos centrais do LangChain são os **Agents** e os **Tools**:

## <font color="#c3d69b">1. Agents</font>

São componentes (ou “agentes”) que utilizam LLMs para **decidir de forma autônoma** quando e como interagir com diferentes ferramentas, a fim de resolver uma tarefa complexa.

Ao receber uma tarefa, o agent analisa o pedido, determina que tipo de ação precisa tomar (por exemplo, consultar uma API, efetuar um cálculo, buscar informação em um banco de dados) e então chama as ferramentas apropriadas para cumprir a meta.

 Por padrão, LangChain implementa o padrão **ReAct** (Reason + Act), em que o modelo elabora seu raciocínio passo a passo, escolhe a ferramenta certa (Act) e usa o resultado para continuar sua linha de raciocínio.

**Exemplo**

 Você tem um agent que precisa responder a uma pergunta sobre o clima. Ele pode consultar uma ferramenta de previsão do tempo (Weather API). Com base na resposta, redige a mensagem final para o usuário.

 Os **Agents** são então “intérpretes inteligentes” que coordenam o uso de ferramentas adequadas, com base no raciocínio fornecido por um modelo de linguagem.

## <font color="#c3d69b">2. Tools</font>

**Tools** (ferramentas) são **interfaces** específicas que o agent pode chamar para executar tarefas ou obter dados externos.
Podem ser funções Python (ex.: realizar cálculos, manipular arquivos), chamadas de APIs (ex.: busca na internet, acesso a banco de dados) ou integrações com outros serviços.

 No LangChain, cada tool é registrada com um nome e uma descrição de uso, para que o LLM entenda **quando** e **por que** chamá-la.
 Ao implementar uma tool, você define como ela recebe parâmetros e qual o formato do retorno.

 **Exemplo**

 Uma tool chamada `search_tool` que acessa uma API de busca e retorna um texto com resultados.
 Uma tool `calculator_tool` que avalia expressões matemáticas.
 Um “retriever_tool” que busca passagens relevantes em um conjunto de documentos usando embeddings.

**Tools**: São funcionalidades (APIs, funções) disponibilizadas ao agent para que ele possa **estender** suas capacidades além do puro texto, acessando dados e realizando ações específicas.

## 3. Como Interagem?

 **Fluxo Típico**

- O usuário faz uma pergunta para o agent.
- O LLM **raciocina** (prompt interno, com “chain-of-thought”), decide se precisa de ajuda externa.
- Seleciona a tool que melhor resolve a necessidade.
- Chama a tool e recebe a resposta.
- Continua o raciocínio com base na informação recebida.
- Retorna uma resposta final ao usuário.

 **Benefícios**

- **Escalabilidade**: você adiciona novas ferramentas sem precisar modificar toda a lógica do agent, pois o LLM aprende quando chamá-las.
- **Modularidade**: cada tool é isolada e tem uma responsabilidade clara.
- **Automação**: o agent gerencia a complexidade de orquestrar várias chamadas de ferramentas.

# <font color="#fac08f">1. Tools</font>

[LangChain Tools](https://python.langchain.com/v0.2/docs/integrations/tools/)

Na própria plataforma da LangChain é possível encontrar algumas tools disponíveis para aplicação, vale ressaltar que algumas delas são pagas para adquirir a chave de alguma API.

Existem diversas tools interessantes disponíveis dentre elas destaco:
- [DuckDuckGo Search]([DuckDuckGo Search | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/integrations/tools/ddg/))
- [Spark SQL Toolkit]([Spark SQL Toolkit | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/integrations/tools/spark_sql/))
- [YouTube]([YouTube | 🦜️🔗 LangChain](https://python.langchain.com/v0.2/docs/integrations/tools/youtube/))

## 1.1 QuickStart Tools

<font color="#fac08f">Lembrando que um tool não é uma IA. </font>

### `DuckDuckGoSearch`

```
from langchain_community.tools import DuckDuckGoSearchRun

ddg_search = DuckDuckGoSearchRun()

search_result = ddg_search.run('Quem foi Alan Turing?')
```

### `PythonREPL`

```
from langchain_experimental.utilities import PythonREPL

python_repl = PythonREPL()
result = python_repl.run('print(5*5)')
print(result)
```

### `WikipediaQueryRun`


```
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


wikipedia = WikipediaQueryRun(
    api_wrapper = WikipediaAPIWrapper(
        lang='pt'
    )
)


wikipedia_results = wikipedia.run('Quem foi Alan Turing?')
print(wikipedia_results)
```


# <font color="#ff0000">2. Agents</font>

<font color="#d99694">React agent</font> é um agente que recebe varias tools e ele mesmo toma a decisão de qual tool deve usar para executar uma função especifica.

## Agent wikipedia Search Tool


```
from langchain_openai import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain.prompts import PromptTemplate


import os
from dotenv import load_dotenv


load_dotenv(dotenv_path=".env")
openai_api_key = os.getenv("OPENAI_API_KEY")


model = ChatOpenAI(
    api_key=openai_api_key,
    model='gpt-3.5-turbo'
)


wikipedia_tool = WikipediaQueryRun(
    api_wrapper = WikipediaAPIWrapper(
        lang='pt'
    )
)


agent_executor = create_python_agent(
    llm=model,
    tool=wikipedia_tool,
    verbose=True, # visualizar a sequencia logica do agente
)


prompt_template = PromptTemplate(
    input_variables=['query'],
    template='''
        Pesquise na web sobre {query} e forneca um resumo sobre o assunto.
        '''
)


query = 'Alan Turing'
prompt = prompt_template.format(query=query)


response = agent_executor.invoke(prompt)
print(response.get('output'))
```

## Agent Python REPL Tool


```
from langchain_openai import ChatOpenAI
from langchain_experimental.utilities import PythonREPL
from langchain.agents import Tool
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain.prompts import PromptTemplate


import os
from dotenv import load_dotenv


load_dotenv(dotenv_path=".env")
openai_api_key = os.getenv("OPENAI_API_KEY")


model = ChatOpenAI(
    api_key=openai_api_key,
    model='gpt-3.5-turbo'
)


python_repl = PythonREPL()
python_repl_tool = Tool(
    name='Python REPL',
    description='Um shell python. Use isso para executar código Python. Execute apenas códigos Python validos.'
                'Se você precisar obter o retorno do código, use a funcão "print(...)".',
    func=python_repl.run,
)

  
agent_executor = create_python_agent(
    llm=model,
    tool=python_repl,
    verbose=True, # visualizar a sequencia logica do agente
)


prompt_template = PromptTemplate(
    input_variables=['query'],
    template='''
        Resolva o calculo {query}.
        '''
)


query = r'e^(-3)'
prompt = prompt_template.format(query=query)


response = agent_executor.invoke(prompt)
print(response.get('output'))
```

## Agent assistente financeiro


```
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.utilities import PythonREPL
from langchain import hub


import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
openai_api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(
    api_key=openai_api_key,
    model='gpt-3.5-turbo'
)


prompt = '''
        Como assistente financeiro pessoal que responderá as perguntas dando dicas financeiras e de investimento.
        Responda tudo em português brasileiro.
        Perguntas {q}
'''

prompt_template = PromptTemplate.from_template(prompt)

python_repl = PythonREPL()

python_repl_tool = Tool(
    name='Python REPL',
    description='Um shell Python. Use isso para executar código Python. Execute apenas códigos Python válidos.'
                'Se você precisar obter o resultado use a funcao print(...).'
                'Use para realizar calculo financeiros necessários para dar dicas.',
    func=python_repl.run()
)

search = DuckDuckGoSearchRun()

duckduckgo_tool = Tool(
    name='Busca DuckDuckgo',
    description='Útil para encontrar informacões, dicas de economia e opcões de investimento.'
                'Você sempre deve pesquisar na internet as melhores dicas usando esta ferramenta.'
                'Não responda diretamente, sua resposta deve informar que há elementos pesquisados na internet.',
    func=search.run,
)


react_instructions = hub.pull("hwchase17/react")
print(react_instructions)


tools = [python_repl_tool, duckduckgo_tool]

agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=react_instructions,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)


question = '''
        Minha renda é de R$10000,00 por mês, o total das minhas despesas é de R$8500,00 mais R$1000,00 de aluguel.
        Quais dicas voce pode me dar?
        '''
output = agent_executor.invoke(
    {
        'input': prompt_template.format(q=question)
    }
)

print(output.get('output'))
```


#### ``hub.pull("hwchase17/react")``

Foi utilizada a lib [LangChain Hub](https://smith.langchain.com/hub/) nela é possível utilizar prompts públicos disponibilizados pela comunidade. No exemplo utilizamos uma hub para react agents, esse prompt foi feito para entregar ao modelo quais tools ele deve usar.


```
react_instructions = hub.pull("hwchase17/react")

print(react_instructions)
```

É possível observar um lista de comportamento que o agente deve seguir, os comandos são genéricos e servem para a maioria dos projetos envolvendo React agents.


## Agent com banco de dados SQL


```
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate


import os
from dotenv import load_dotenv


load_dotenv(dotenv_path=".env")
openai_api_key = os.getenv("OPENAI_API_KEY")


model = ChatOpenAI(
    api_key=openai_api_key,
    model='gpt-4'
)


db = SQLDatabase.from_uri('sqlite:///ipca.db')

toolkit = SQLDatabaseToolkit(
    db=db,
    model=model,
)


system_message = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=model,
    tools=toolkit.get_tools(),
    prompt=system_message,
)


agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=True,
)


prompt= '''
        Use as ferramentas necessárias para responder perguntas relacionadas ao histórico de IPCA ao longo dos anos.
        Responda tudo em portugues brasileiro.
        Perguntas: {q}
        '''

  

prompt_template = PromptTemplate.from_template(prompt)

question = 'Qual o mês e ano tiveram o maior IPCA?'

output = agent_executor.invoke(
    {
    'input': prompt_template.format(q=question)
    }
) 

print(output.get('output'))
```








---
# Contexto
- **Situação**: Primeiro semestre no ita e fazendo o curso do Pycode BR.
- **Fonte**: 

## Próximos Passos
- **Ação 1**: 
- **Ação 2**: 

## Referências
- [LangChain](https://www.langchain.com/) - Site Oficial
- [OpenAi API](https://platform.openai.com/api-keys) - site da OpenAI para devs
- [LangChang Docs](https://python.langchain.com/v0.2/docs/introduction/)
- [LangChain Community](https://pypi.org/project/langchain-community/)
- [LangChain Expression Language](https://python.langchain.com/v0.2/docs/how_to/#langchain-expression-language-lcel)
- [How to Chain runnables](https://python.langchain.com/v0.2/docs/how_to/sequence/)
- [LangChain Tools](https://python.langchain.com/v0.2/docs/integrations/tools/)
- [LangChain Hub](https://smith.langchain.com/hub/)
- [SQLDatabase Toolkit](https://python.langchain.com/v0.2/docs/integrations/tools/sql_database/)
=== gpt ===
-  [Gpt key](https://platform.openai.com/playground/chat?models=gpt-4o) - plataforma para desenvolvedores
- [Api Pricing](https://openai.com/api/pricing/)
- [Api key](https://platform.openai.com/api-keys)
- [models](https://platform.openai.com/docs/models)



