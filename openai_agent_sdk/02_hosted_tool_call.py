#---------------------- TOOLS ----------------------#
# Here we define the tools that the agent can use.
# Hosted tools
#    - These run on LLM servers alongside the AI model.
#    - OpenAI offers retrieval, web search, and computer use as hosted tools.
# Function-calling tools
#    - You define the Python functions, and the agent can call them as a tool.
#    - Useful for tasks that require specific local context or capabilities.
# Agents as tools
#    - Agents can use other agents as tools.
#    - Allowing Agent to call other agent without handing off to them.

from dotenv import load_dotenv
from agents import Agent, Runner
from agents import WebSearchTool
load_dotenv()

hello_agent = Agent(
    name="Hello World Agent",  
    tools=[WebSearchTool()],
)

result = Runner.run_sync(hello_agent, "What is on OpenAI's homepage?")

print(result.final_output)

