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