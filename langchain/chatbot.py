from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
    # other params...
)

messages = [
    ("system", "You are a very chatty and friendly bot. And you never take anything serious."),
]

while True:
    user_input = input("You: ")
    normalized = user_input.strip().lower()
    
    if any(word in normalized for word in ("bye", "thank you", "exit")):
        break
    messages.append(("user",user_input))
    response = model.invoke(messages)
    print(response.content)
    messages.append(("assistant",response.content))
