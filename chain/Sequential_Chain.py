from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b"
)

prompt1 = PromptTemplate(
    template="Generate details report this {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Provide the 5 important lines about this \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | llm | parser | prompt2 | llm | parser

print(chain.invoke({
    "topic" : "cricket"
}))

print(chain.get_graph().print_ascii())