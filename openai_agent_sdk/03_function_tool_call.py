import requests
import asyncio
from typing_extensions import TypedDict, Any
from agents import Agent, Runner, FunctionTool, function_tool
from dotenv import load_dotenv
load_dotenv()


@function_tool
async def get_weather(location: str) -> str:
    """Get the current weather for a given location.
    Args:
        location (str): The location to get the weather for.
    """
    url = f"https://wttr.in/{location.lower()}"
    response = requests.get(url)
    if  response.status_code == 200:
        return f"The weather in {location} is {response.text.strip()}."
    return f"Something went wrong"

agent = Agent(
    name="Weather Agent",
    instructions="You are a helpful agent.",
    tools=[get_weather],
)


async def main():
    result = await Runner.run(agent, input="Tell me what's the weather in Tokyo?")
    print(result.final_output)
    # The weather in Tokyo is sunny.


if __name__ == "__main__":
    asyncio.run(main())