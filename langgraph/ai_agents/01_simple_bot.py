from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START,END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatOpenAI(model="gpt-4o")

def process(state: AgentState)-> AgentState:
    response = llm.invoke(state['messages'])
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("processor",process)
graph.add_edge(START,"processor")
graph.add_edge("processor",END)
agent = graph.compile()

# response =  agent.invoke("hello, how are you?")
user_msg = input("Enter: ")
response = agent.invoke({"messages": [HumanMessage(content=user_msg)]})
print(response)

user_input = input("Enter:")
while user_input.lower() != "exit":
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})
    print(response)
    user_input = input("Enter: ")