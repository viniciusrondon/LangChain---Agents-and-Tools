from langchain_experimental.utilities import PythonREPL

python_repl = PythonREPL()
result = python_repl.run('print(5*5)')
print(result)