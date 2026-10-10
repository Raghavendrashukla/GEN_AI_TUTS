"""
Chain का मतलब होता है — कई सारे स्टेप्स को एक के बाद एक जोड़ देना, ताकि पूरा काम एक ही बार में हो जाए।
सबसे आधुनिक तरीका है LCEL (LangChain Expression Language), जिसमें | (pipe) का इस्तेमाल होता है।
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b"
)

prompt = PromptTemplate(
    input_variables=['text'],
    template="Generate 5 lines about {text}"
)

output_parser = StrOutputParser()

# 1. बिना Chain के (पुराना / मैन्युअल तरीका)
response = prompt.invoke({
    "text" : "cricket"
})

result = llm.invoke(response)
print("Without using chain____________________________")
print(result.content)

# 2. Chain के साथ (आधुनिक और सही तरीका)
chain = prompt | llm | output_parser

print(chain.invoke({
    "text" : "cricket"
}))

print(chain.get_graph().print_ascii()) # it will show how to build our chain 