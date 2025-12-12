from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# Annotated - provides additional context without affecting the type itself
# Sequence - to automatically handle the state updates for sequences such as by adding new messages to a chat history
# Reducer Function 
#        - Rule that controls how updates from nodes are combined with the existing state.
#        - Tell us how to merge new data into current state.
#        - Without a reducer, updates would have replaced the existing value entirely!

class AgentState(TypedDict):
    """State of the agent, containing messages."""
    messages: Annotated[Sequence[BaseMessage],add_messages]

@tool
def add(a:int, b:int):
    """This is an addition function that add two numbers"""
    return a + b

@tool
def subtract(a:int, b:int):
    """This is an addition function that add two numbers"""
    return a - b

@tool
def multiply(a:int, b:int):
    """This is an addition function that add two numbers"""
    return a * b

tools = [add, subtract, multiply]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my AI assistant, please answer my query to the best of your abilitiy.")
    response = model.invoke([system_prompt]+ state['messages'])
    return {"messages":[response]}

def should_continue(state: AgentState) -> AgentState:
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return 'end'
    else: 
        return 'continue'
    
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)
tool_node = ToolNode(tools=tools)
graph.add_node('tools', tool_node)
graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    source="our_agent",
    path= should_continue,
    path_map={
        # edge : node 
        'continue' : 'tools',
        'end': END
    },
)
graph.add_edge("tools","our_agent") 
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s['messages'][-1]
        if isinstance(message,tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages":[("user","Add 40 + 12 and then multiply the result by 6. And add 125 + 6547, then subtract it wilth 2321")]}
print_stream(app.stream(inputs,stream_mode="values"))