from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

import asyncio

mcp_config = {
    "playwright": {"command": "npx", "args": ["@playwright/mcp@latest"], "transport": "stdio"},
    "time": {"command": "uvx", "args": ["mcp-server-time", "--local-timezone=America/New_York"], "transport": "stdio"},
}

_client = None


def get_mcp_client():
    """Get or initialize MCP client lazily to avoid issues with LangGraph Platform."""
    global _client
    if _client is None:
        _client = MultiServerMCPClient(mcp_config)
    return _client


async def get_mcp_tools():
    """Retrieve available tools from MCP client.
    Asynchronous version to accommodate potential async initialization.
    """
    client = get_mcp_client()
    mcp_tools = (
        await client.get_tools()
    )  # asyncio.run(client.get_tools()) # .await if asyncio.iscoroutine(client.get_tools()) else client.get_tools()
    return mcp_tools


def get_playwright_tool():
    """Get the Playwright tool from MCP tools."""
    mcp_server_config = {
        "playwright": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-playwright"],
            "transport": "stdio",
        }
    }
    playwright_tools = load_mcp_tools(mcp_config=mcp_server_config)
    return playwright_tools
