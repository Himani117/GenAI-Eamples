from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START,END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatOpenAI(model="gpt-4o")

def process(state: AgentState) -> AgentState:
    """This node will do solve the request of human"""
    response = llm.invoke(state["messages"])

    state['messages'].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("processor", process)
graph.add_edge(START, "processor")
graph.add_edge("processor", END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    response = agent.invoke({"messages":conversation_history})
    print(response['messages'])
    conversation_history = response['messages']
    user_input = input("\nEnter: ")

with open("conversation_history.txt", "w") as f:
    f.write("Your conversation log:\n")
    for message in conversation_history:
        if isinstance(message,HumanMessage):
            f.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n\n")
    f.write("\nEnd of conversation log.\n")

    print("Conversation history saved to conversation_history.txt")