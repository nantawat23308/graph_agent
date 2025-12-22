from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from src.agent_deep_research import deep_researcher_graph, deep_researcher_builder
from langgraph.config import RunnableConfig
from src.utility import tavily_search, think_tool
from src.agent_rag import rag_agent
from src.agent_normal import AgenticTOOL, get_stock_price
from src.mcp_client.client import get_mcp_tools
import uuid
import asyncio

# Tools Milvus
from src.langchain_milvus.tools_utils import milvus_search, rag_milvus

# from PIL import Image, ImageDraw


async def deep_research():
    model = init_chat_model("bedrock_converse:us.meta.llama4-maverick-17b-instruct-v1:0")
    tools = [tavily_search, think_tool]
    thread_id = uuid.uuid4()
    config: RunnableConfig = {
        "configurable": {"llm": model, "allow_clarification": False, "researcher_tools": tools, "thread_id": thread_id},
        "recursion_limit": 50,
    }

    result = await deep_researcher_graph.ainvoke(
        input={
            "messages": [
                "Can you provide an in-depth analysis of the impact of climate change on global agriculture?",
                "I would like to understand both the challenges and potential solutions.",
            ]
        },
        config=config,
    )

    print("Final Result:")
    print(result["messages"][-1].content)

    print(result['messages'])


def rag_milvus_agent():
    thread_id = uuid.uuid4()
    model = init_chat_model("bedrock_converse:us.meta.llama4-maverick-17b-instruct-v1:0")
    config: RunnableConfig = {"configurable": {"llm": model, "thread_id": thread_id}, "recursion_limit": 100}
    inputs = {"question": "What is Attention?"}
    output = rag_agent.invoke(
        inputs,
        config=config,
    )
    print("Final Answer:")
    print(output.get("generation"))


async def normal_agent():
    sample_2 = "What is the current stock price of Tesla?"
    model = init_chat_model("bedrock_converse:us.meta.llama4-maverick-17b-instruct-v1:0")
    tools = [
        tavily_search,
        get_stock_price,
    ] + await get_mcp_tools()
    result = await AgenticTOOL(llm=model, tools=tools).run(user_input=sample_2)
    print("Final Result:")
    print(result)


if __name__ == "__main__":
    asyncio.run(deep_research())
    # rag_milvus_agent()
    # asyncio.run(normal_agent())
