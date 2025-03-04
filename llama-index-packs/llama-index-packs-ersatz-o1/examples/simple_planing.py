from llama_index.llms.openai import OpenAI

from llama_index.packs.ersatz_o1 import ErsatzO1QueryEngine

llm = OpenAI(model="gpt-4-turbo")

task_context = """
Tim gets a promotion that offers him a 5% raise on his $20000 a month salary. It also gives him a bonus worth half a
month’s salary.
"""

query_engine = ErsatzO1QueryEngine(
    context=task_context,
    llm=llm,
    reasoning_paths=5,
    verbose=True,
)

if __name__ == "__main__":
    res = query_engine.query("How much money will he make in a year?")
    print(res)
