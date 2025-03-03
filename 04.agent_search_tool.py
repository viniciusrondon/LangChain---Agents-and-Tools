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