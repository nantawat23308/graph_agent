from typing_extensions import Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from langchain_core.runnables import RunnableConfig

from src.deep_research.state import ResearcherState, ResearcherOutputState
from src.utility import get_today_str
from src.prompts.prompt_research import (
    research_agent_prompt,
    compress_research_system_prompt,
    compress_research_human_message,
)
from src.configuration import Configuration
from src.logger import log


def llm_call(state: ResearcherState, config: RunnableConfig):
    """Analyze current state and decide on next actions.

    The model analyzes the current conversation state and decides whether to:
    1. Call search tools to gather more information
    2. Provide a final answer based on gathered information

    Returns updated state with the model's response.
    """
    log.debug("--- LLM CALL ---")
    configurable = Configuration.from_runnable_config(config)
    model_with_tools = (
        configurable.get_model()
        .bind_tools(configurable.get_research_tools())
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    ## Inject System prompt with Available Tools and Instructions
    research_prompt = research_agent_prompt.format(
        date=get_today_str(), tool_instructions=configurable.get_tool_instructions()
    )
    result_llm_call = model_with_tools.invoke([SystemMessage(content=research_prompt)] + state["researcher_messages"])
    log.debug("--- LLM Success ---")
    return {"researcher_messages": [result_llm_call]}


def tool_node(state: ResearcherState, config: RunnableConfig):
    """Execute all tool calls from the previous LLM response.

    Executes all tool calls from the previous LLM responses.
    Returns updated state with tool execution results.
    """
    configurable = Configuration.from_runnable_config(config)
    tools = configurable.get_research_tools()
    tools_by_name = {tool.name: tool for tool in tools}

    tool_calls = state["researcher_messages"][-1].tool_calls
    log.debug("--- TOOL NODE: Executing %d tool calls ---", len(tool_calls))

    # Execute all tool calls
    observations = []
    for tool_call in tool_calls:
        log.debug("Invoking tool: %s with args: %s", tool_call["name"], tool_call["args"])
        tool = tools_by_name[tool_call["name"]]
        observations.append(tool.invoke(tool_call["args"]))

    # Create tool message outputs
    tool_outputs = [
        ToolMessage(content=observation, name=tool_call["name"], tool_call_id=tool_call["id"])
        for observation, tool_call in zip(observations, tool_calls)
    ]

    return {"researcher_messages": tool_outputs}


def compress_research(state: ResearcherState, config: RunnableConfig) -> dict:
    """Compress research findings into a concise summary.

    Takes all the research messages and tool outputs and creates
    a compressed summary suitable for the supervisor's decision-making.
    """
    log.debug("--- COMPRESS RESEARCH ---")
    configurable = Configuration.from_runnable_config(config)
    compress_model = configurable.get_model().with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    system_message = compress_research_system_prompt.format(date=get_today_str())
    messages = (
        [SystemMessage(content=system_message)]
        + state.get("researcher_messages", [])
        + [HumanMessage(content=compress_research_human_message)]
    )
    response = compress_model.invoke(messages)

    # Extract raw notes from tool and AI messages
    raw_notes = [str(m.content) for m in filter_messages(state["researcher_messages"], include_types=["tool", "ai"])]

    return {"compressed_research": str(response.content), "raw_notes": ["\n".join(raw_notes)]}


def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """Determine whether to continue research or provide final answer.

    Determines whether the agent should continue the research loop or provide
    a final answer based on whether the LLM made tool calls.

    Returns:
        "tool_node": Continue to tool execution
        "compress_research": Stop and compress research
    """
    messages = state["researcher_messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, continue to tool execution
    if last_message.tool_calls:
        return "tool_node"
    # Otherwise, we have a final answer
    return "compress_research"


research_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState, context_schema=Configuration)
research_builder.add_node("llm_call", llm_call)
research_builder.add_node("tool_node", tool_node)
research_builder.add_node("compress_research", compress_research)

research_builder.add_edge(START, "llm_call")
research_builder.add_conditional_edges(
    "llm_call", should_continue, {"tool_node": "tool_node", "compress_research": "compress_research"}
)
research_builder.add_edge("tool_node", "llm_call")
research_builder.add_edge("compress_research", END)

researcher_agent = research_builder.compile(checkpointer=True)
