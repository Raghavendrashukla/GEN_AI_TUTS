from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b"
)

messages = [
    ("system", "you are helpful assistant"),
    ("human", "I love programming")
]

ai_msg = llm.invoke(messages)
print(ai_msg.content)