from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from src.agent_top import deep_researcher_graph, deep_researcher_builder
from langgraph.config import RunnableConfig
from src.utility import tavily_search, think_tool
from PIL import Image, ImageDraw

async def main():
    checkpointer = InMemorySaver()
    model = init_chat_model("bedrock_converse:us.meta.llama4-maverick-17b-instruct-v1:0")
    tools = [tavily_search, think_tool]
    # config = Configuration(
    #     llm=model,
    #     researcher_tools=tools,
    # )
    config: RunnableConfig = {
        "configurable": {
            "llm": model,
            "allow_clarification": False
        },
        "recursion_limit": 50
    }
    full_agent = deep_researcher_builder.compile(checkpointer=checkpointer)
    # image = full_agent.get_graph(xray=True).draw_mermaid_png()
    # PILimage = Image.fromarray(image)
    # PILimage.save('result.png')

    result = await deep_researcher_graph.ainvoke(
        input={
            "messages": [
                "Can you provide an in-depth analysis of the impact of climate change on global agriculture?",
                "I would like to understand both the challenges and potential solutions."
            ]
        },
        config=config,

    )

    print("Final Result:")
    print(result['messages'])

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())