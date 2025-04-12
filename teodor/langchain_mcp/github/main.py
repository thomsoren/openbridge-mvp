import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_openai import ChatOpenAI


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("OPENAI_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found in environment variables")
        return self.api_key


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]}
            if self.config.get("env")
            else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)

                return result

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self, name: str, description: str, input_schema: dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class LLMClient:
    """LangChain-powered OpenAI client with tool handling."""
    
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-4-turbo",  # or "gpt-4o" when available
            temperature=0.7,
            max_tokens=4096
        )
        self.parser = StrOutputParser()
        
    def get_response(self, messages: list[dict[str, str]]) -> str:
        # Convert message format for LangChain
        lc_messages = [
            ("system", messages[0]["content"]),
            *[(msg["role"], msg["content"]) for msg in messages[1:]]
        ]
        
        prompt = ChatPromptTemplate.from_messages(lc_messages)
        chain = prompt | self.llm | self.parser
        
        try:
            return chain.invoke({})
        except Exception as e:
            logging.error(f"LangChain error: {str(e)}")
            return f"LLM Error: {str(e)}"



from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List, Dict, Union

class LLMClient:
    """LangChain-powered OpenAI client with proper template handling."""
    
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-4-turbo",  # Using valid model name
            temperature=0.7,
            max_tokens=4096
        )
        self.parser = StrOutputParser()
        
    def get_response(self, messages: List[Dict[str, str]]) -> str:
        try:
            # Convert to LangChain message format
            lc_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    lc_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    lc_messages.append(AIMessage(content=msg["content"]))
            
            # Create prompt with proper variable handling
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=messages[0]["content"]),  # System instruction
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
            ])
            
            chain = prompt | self.llm | self.parser
            return chain.invoke({
                "chat_history": lc_messages[1:-1],  # All messages except first and last
                "input": lc_messages[-1].content   # Last message is the current input
            })
            
        except Exception as e:
            logging.error(f"LLM Error: {str(e)}")
            return "I encountered an error processing your request."

class ChatSession:
    """Updated with proper prompt template handling."""
    def __init__(self, servers: list[Server], llm_client: LLMClient) -> None:
        self.servers: list[Server] = servers
        self.llm_client: LLMClient = llm_client
    
    async def get_tools_prompt(self) -> str:
        all_tools = []
        for server in self.servers:
            tools = await server.list_tools()
            all_tools.extend(tools)
        
        tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])
        
        return f"""You are a Git repository assistant with these available tools:
    {tools_description}

    For Git operations, ALWAYS use the exact JSON format:
    {{
        "tool": "git-list-files",  # or other git-* commands
        "arguments": {{
            "path": "directory_path",  # or other required parameters
            "branch": "branch_name"    # when applicable
        }}
    }}

    Rules:
    1. NEVER add commentary outside the JSON
    2. Ask for clarification if the request is ambiguous
    3. Default to 'main' branch if none specified
    4. Use "./" for repository root
    """


    async def start(self) -> None:
        """Main chat session handler with proper prompt structure."""
        try:
            # Initialize servers
            for server in self.servers:
                await server.initialize()

            # Create initial messages
            messages = [{
                "role": "system",
                "content": await self.get_tools_prompt()
            }]

            while True:
                try:
                    user_input = input("You: ").strip()
                    if user_input.lower() in ["quit", "exit"]:
                        break
                    messages.append({"role": "user", "content": user_input})

                    # Get LLM response
                    llm_response = self.llm_client.get_response(messages)
                    
                    # Try to parse as JSON tool call
                    try:
                        tool_data = json.loads(llm_response.strip())
                        if isinstance(tool_data, dict) and "tool" in tool_data:
                            result = await self.process_llm_response(llm_response)
                            print(f"Tool Result: {result}")
                            continue
                    except json.JSONDecodeError:
                        pass
                    
                    # If not JSON, show direct response
                    print(f"Assistant: {llm_response}")

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {str(e)}")
                    continue

        finally:
            await self.cleanup_servers()


async def main() -> None:
    """Initialize and run the chat session."""
    config = Configuration()
    server_config = config.load_config("servers_config.json")
    # Debugging the servers list
    servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]

    for server in servers:
        print(f"Server Name: {server.name}, Config: {server.config}")
    llm_client = LLMClient(config.llm_api_key)
    chat_session = ChatSession(servers, llm_client)
    await chat_session.start()


if __name__ == "__main__":
    asyncio.run(main())