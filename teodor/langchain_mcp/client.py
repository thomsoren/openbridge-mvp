import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI
# math_server.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Get the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")


model = ChatOpenAI(model="gpt-4o-mini")

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["/home/teodorrk/projects/openbridge-mvp/teodor/langchain_mcp/math_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
            print(agent_response)

# Run the async function
asyncio.run(main())