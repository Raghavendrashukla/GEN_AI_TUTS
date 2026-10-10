"""
Parallel Chain का मतलब है — एक ही इनपुट को एक साथ कई Chains को देना, और सभी का रिजल्ट एक साथ लेना।
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b"
)

# अलग-अलग Prompts
summary_prompt = ChatPromptTemplate.from_template(
    "give me the summary of this {topic}"
)

points_prompt = ChatPromptTemplate.from_template(
    "Tell me 3 main points of this {topic}"
)

simple_prompt = ChatPromptTemplate.from_template(
    "please understand me as a simple language in this {topic}"
)

# Parallel Chain बनाई
parallel_chain = RunnableParallel({
    "summary": summary_prompt | llm | StrOutputParser(),
    "main_points": points_prompt | llm | StrOutputParser(),
    "simple_explanation": simple_prompt | llm | StrOutputParser()
})

# इस्तेमाल
result = parallel_chain.invoke({"topic": "Langchain"})

print(result)
print(parallel_chain.get_graph().print_ascii())