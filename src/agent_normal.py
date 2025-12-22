import asyncio
from typing import Annotated

import yfinance as yf
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from src.logger import log
from src.mcp_client.client import get_mcp_tools

load_dotenv()

SYSTEM_PROMPT = """<role>You are a high-precision Autonomous Assistant equipped with specialized tools.
        Your goal is to answer user queries accurately, using tools only when necessary.
        </role>
        
<tool_protocol>
1. SINGLE EXECUTION: For any specific data point, call the relevant tool exactly ONCE.
2. TRUST OUTPUT: Treat all data returned by a tool as final and accurate. Do not attempt to "verify" or "refresh" the same data point within the same conversation turn.
3. DATA SUFFICIENCY: Once you have received a tool response that addresses the user's request, you must stop calling tools and immediately provide your final answer to the user.
4. NO REDUNDANCY: Do not call the same tool with the same parameters more than once in a single execution chain.
</tool_protocol>
    
<instructions>:
1. Analyze the user's question carefully.
2. Determine if any tools are needed to gather information.
3. Use the tools effectively to obtain relevant data.
4. Synthesize the information and provide a clear, concise response.
</instructions>

<response_guidelines>
- Transition immediately from tool-use to final response once the required information is obtained.
- If a tool returns an error or "not found," report that to the user instead of retrying indefinitely.
- Provide direct, factual answers. Eliminate introductory phrases like "I have found the information...
</response_guidelines>

<fallback>
If you cannot obtain the necessary information through tools, respond with:
```"not found".```
Critical: Do not attempt to fabricate or guess answers. Always rely on tool outputs.
</fallback>
    
<response_format>
You must output the data in the exact JSON structure.
Output only raw JSON. Do not explain anything. Do not include Markdown or headers.
ALWAYS follow the structure and output format exactly. NEVER provide commentary, summaries, or explanations. Your only output is valid JSON or {{"status": "no_valid_entries"}}.
Add a rule enforcing no Markdown, no YAML, no formatting hints—pure JSON only.
DO NOT use Markdown, YAML, formatting hints, or explanations. Output must be strict raw JSON only — no headings, no text commentary, no quotes around string values unless required by JSON syntax. Any output that violates this rule is invalid.


{{
  "status": "success",
  "answer": "<Your concise, factual answer here>",
  "data_sources": ["<List of data sources used, if any>"]
}}
</response_format>
"""

tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
)


@tool
def get_stock_price(ticker: Annotated[str, "The uppercase stock symbol, e.g. 'AMZN'"]) -> str:
    """
    Retrieves the latest market price for a specific stock ticker.
    The input must be a valid ticker symbol (e.g., 'AAPL', 'NVDA', 'BTC-USD').
    Args:
        ticker: The stock ticker symbol to look up.
    Returns:
        A string containing the current stock price or an error message.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        currency = info.get("currency", "USD")

        if price:
            return f"The current price of {ticker} is {price} {currency}."
        else:
            return f"Could not find the current price for {ticker}."
    except Exception as e:
        return f"Error fetching stock price for {ticker}: {str(e)}"


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: ({str(e)})",
            tool_call_id=request.tool_call["id"],
        )


@wrap_tool_call
async def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return await handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: ({str(e)})",
            tool_call_id=request.tool_call["id"],
        )


class AnswerResponse(BaseModel):
    status: str = Field(..., description="Status of the response, e.g., 'success' or 'no_valid_entries'")
    answer: str = Field(..., description="The concise, factual answer to the user's query")
    data_sources: list[str] = Field(..., description="List of data sources used, if any")


class AgenticTOOL:
    def __init__(self, llm, tools):
        self.model = llm
        self.system_prompt = SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                },
            ]
        )
        self.agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=self.system_prompt,
            middleware=[
                handle_tool_errors,
            ],
            # response_format=ToolStrategy(AnswerResponse)
        )

    async def run(self, user_input: str) -> str:
        response_agent = await self.agent.ainvoke(
            {"messages": [HumanMessage(user_input)]},
        )
        # for step in self.agent.stream(
        #     {"messages": user_input},
        #     stream_mode="values",
        # ):
        #     step["messages"][-1].pretty_print()
        log.debug(response_agent["messages"][-1].content)
        if "structured_response" in response_agent:
            return response_agent["structured_response"]  # result["messages"][-1].content
        return response_agent["messages"][-1].content


async def main():
    sample = "What is the torque for the M5 cylinder head?"
    sample_2 = "What is the current stock price of Tesla?"
    sample_3 = "Who is president of America in 2025?"
    sample_4 = "What is today's date?"
    model = init_chat_model("bedrock_converse:us.meta.llama4-maverick-17b-instruct-v1:0", temperature=0.1)
    tools = [
        # tavily_search_tool,
        get_stock_price,
    ] + await get_mcp_tools()
    result = await AgenticTOOL(llm=model, tools=tools).run(user_input=sample_2)
    print("Final Result:")
    print(result)


if __name__ == "__main__":
    # model = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash",
    #     temperature=1.0,
    #     max_tokens=None,
    #     timeout=None,
    #     max_retries=2,
    # )
    asyncio.run(main())
