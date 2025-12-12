from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import json_loader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embeddings)
pdf_path = "xyz.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

page_split = text_splitter.split_documents(pages)
_ = vector_store.add_documents(page_split)

retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 5} # K is the amount of chunks to return
)

@tool 
def retriever_tool(query:str) -> str:
    """ 
    This tool searches and returns the information from the document.
    """

    docs = retriever.invoke(query)

    if not docs:return "I found no relevant information in the document."

    results = []
    for i, docs in enumerate(docs):
        results.append(f"Document {i}:\n{docs.page_content}")

    return "\n\n".join(results)

tool = [retriever_tool]
llm = llm.bind_tools(tool)

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result,'tool_calls') and len(result.tool_calls) > 0

system_prompt = """ 
    You are an intelligent AI Assistant who answer questions about attached document loaded into your knowledge base.
    Use the retriever tool available to answer questions about the attached document. You can make multiple calls if needed.
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    Please always cite the specific parts of the document you use in your answer.
"""

tool_dict = {our_tool.name: our_tool for our_tool in tool} # creating a dictionary of our tool

# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current State"""
    messages = list(state['messages'])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {'messages':[message]}

# Retriever Agent
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response"""
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query','No query provided')}")

        if not t['name'] in tool_dict: # Checks if a valid tool is present
            print(f"\n Tool: {t['name']} does not exist.")
            result = "Incorrect Tool name, please Retry and select tool from List of Available tools."
        
        else:
            result = tool_dict[t['name']].invoke(t['args'].get('query',""))
            print(f"Result length: {len(str(result))}")

        # Append the Tool Message
        results.append(ToolMessage(tool_call_id=t['id'], 
                                   name = t['name'],
                                   content=str(result)))

    print("Tools Execution complete. Back to the model!")
    return {'messages':results}

graph = StateGraph(AgentState)
graph.add_node("llm",call_llm)
graph.add_node("retriever_agent",take_action)

graph.add_conditional_edges(
    source="llm",
    path=should_continue,
    path_map={
        # edge : node
        True: "retriever_agent",
        False: END
    },
)
graph.add_edge("retriever_agent","llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

def running_agent():
    print("\n=== RAG Agent ===")
    while True:
        user_input = input("\n what is your question: ")
        if user_input.lower() in ['exit','quit']:
            break

        messages = [HumanMessage(content=user_input)] # converts back to human message type

        result = rag_agent.invoke({'messages' : messages})

        print("\n === Answer ===")
        print(result['messages'][-1].content)


running_agent()