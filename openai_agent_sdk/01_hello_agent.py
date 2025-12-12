# uv add openai-agents

from dotenv import load_dotenv
from agents import Agent,Runner
load_dotenv()

hello_agent = Agent(
    name="Hello World Agent",  
)

result = Runner.run_sync(hello_agent, "Hey there! How are you?")

print(result.final_output)