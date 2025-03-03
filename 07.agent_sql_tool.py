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