"""
Prompt Template = ek reusable, dynamic prompt jo variables ke saath aata hai. 
Aap ek baar structure banate ho, phir different values daal ke different prompts generate kar sakte ho.
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq( 
    model="openai/gpt-oss-120b"
)

# A. साधारण PromptTemplate (पुराने Completion मॉडल के लिए)
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate(
    template="My name is {name} and i want to discuss about this {topic}",
    input_variables=['name','topic']
)

format = prompt | llm
result = format.invoke({
    "name" : "ABC",
    "topic" : "how to save money"
})
# print(result.content)

# B. ChatPromptTemplate (सबसे ज्यादा इस्तेमाल होता है)
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "Tum ek helfull assistant for mere"),
    ("human", "Mere pass {amount} hain, toh batao mein aur kaise invest kru?")
])

chain = prompt | llm
response = chain.invoke({
    "amount" : 5000
})

# print(response.content)

# C : Example 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

# ─────────────────────────────────────────
# 1. Input Validation (Pydantic)
# ─────────────────────────────────────────
class InvestmentQuery(BaseModel):
    amount: int = Field(..., gt=0, description="investment amount")
    age: int = Field(default=25, ge=18, le=100)
    risk : str = Field(default="medium", pattern="^(low|medium|high)$")

# ─────────────────────────────────────────
# 2. Prompt Template
# ─────────────────────────────────────────
SYSTEM_PROMPT = """You are an experienced financial advisor.
Rules:
- Provide advice based on the user's investment amount, age, and risk profile.
- Always respond in Hindi.
- Mention risk factors.
- Never promise guaranteed returns."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",(
        "i have an {amount}"
        "my age is {age}"
        "i can take a risk"
        "how to invest the amount in india and where?"
    ))
])

def build_chain() -> Runnable:
    llm = ChatGroq( 
        model="openai/gpt-oss-120b",
        temperature=0.7,
        max_tokens=1024,
        timeout=30
    )
    
    return prompt | llm | StrOutputParser()


chain = build_chain()
try : 
    invest = InvestmentQuery(amount=0, age=35, risk="medium")
    response = chain.invoke(invest.model_dump())
    print(response.content)
except Exception as e:
    print("Error : {e}")