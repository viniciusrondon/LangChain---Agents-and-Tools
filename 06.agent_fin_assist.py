from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, create_react_agent, AgentExecutor
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