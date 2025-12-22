from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage, HumanMessage
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_tavily import TavilySearch
from typing import Annotated
import yfinance as yf
from dotenv import load_dotenv
from src.logger import log
load_dotenv()

SYSTEM_PROMPT = """You are a high-precision technical assistant. 
        Use the tools provided to find information. 
        If the tool returns no information, say 'I don't know'.
        Provide direct answers without conversational filler."""
tavily_search_tool = TavilySearch(
        max_results=5,
        topic="general",
    )


@tool
def get_stock_price(ticker: Annotated[str, "The uppercase stock symbol, e.g. 'AMZN'"]) -> str:
    """
    Retrieves the latest market price for a specific stock ticker.
    The input must be a valid ticker symbol (e.g., 'AAPL', 'NVDA', 'BTC-USD').
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


def agent_tool_call(llm, tools, user_input: str):
    """
    Simple agent tool call example
    :param user_input:
    :return:
    """
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    # user_input = "What nation hosted the Euro 2024? Include only wikipedia sources."
    result = agent.invoke(
            {"messages": [HumanMessage(user_input)]},
    )

    # print(result["messages"][-1].content)
    for step in agent.stream(
            {"messages": user_input},
            stream_mode="values",
    ):
        step["messages"][-1].pretty_print()
    log.debug(result["messages"][-1].content)
    return result["messages"][-1].content


def tool_calling_ag(llm, tools, user_input: str):
    """
    Agent with create_tool_calling_agent.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # Set to True to see the "Thinking" process
        handle_parsing_errors=True
    )
    response = agent_executor.invoke({"input": user_input})

    log.debug(response["output"])
    return response["output"]


if __name__ == "__main__":
    model = init_chat_model("bedrock_converse:us.meta.llama4-maverick-17b-instruct-v1:0")
    sample = "What is the torque for the M5 cylinder head?"
    sample_2 = "What is the current stock price of Tesla?"
    tools = [
        # tavily_search_tool,
        get_stock_price
    ]
    agent_tool_call(llm=model, tools=tools, user_input=sample_2)



