import asyncio
from typing import Literal
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command

from src.deep_research.agent_research import researcher_agent

from src.deep_research.state import ConductResearch, ResearchComplete, SupervisorState
from src.configuration import Configuration
from src.utility import think_tool, get_today_str, get_notes_from_tool_calls, refine_draft_report
from src.prompts.prompt_supervisor import lead_researcher_with_multiple_steps_diffusion_double_check_prompt

load_dotenv()


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.

    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.

    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings

    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # logger.debug("Starting supervisor")

    configurable = Configuration.from_runnable_config(config)
    max_researcher_iterations = configurable.max_researcher_iterations
    max_concurrent_research_units = configurable.max_concurrent_research_units

    # Configure the supervisor model with available tools
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    research_model = (
        configurable.get_model()
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    # Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])

    system_message = lead_researcher_with_multiple_steps_diffusion_double_check_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=max_concurrent_research_units,
        max_researcher_iterations=max_researcher_iterations,
    )

    messages = [SystemMessage(content=system_message)] + supervisor_messages

    response = await research_model.ainvoke(messages)

    # Update state and proceed to tool execution
    # logger.debug("Supervisor response generated, moving to supervisor_tools")
    return Command(
        goto="supervisor_tools",
        update={"supervisor_messages": [response], "research_iterations": state.get("research_iterations", 0) + 1},
    )


async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute supervisor decisions - either conduct research or end the process.

    Handles:
    - Executing think_tool calls for strategic reflection
    - Launching parallel research agents for different topics
    - Aggregating research results
    - Determining when research is complete

    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with model settings

    Returns:
        Command to continue supervision, end process, or handle errors
    """
    configurable = Configuration.from_runnable_config(config)
    max_researcher_iterations = configurable.max_researcher_iterations

    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    # Initialize variables for single return pattern
    tool_messages = []
    all_raw_notes = []
    draft_report = ""
    next_step = "supervisor"  # Default next step
    should_end = False

    # Check exit criteria first
    exceeded_iterations = research_iterations >= max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete = any(tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls)

    if exceeded_iterations or no_tool_calls or research_complete:
        should_end = True
        next_step = END

    else:
        # Execute ALL tool calls before deciding next step
        try:
            # Separate think_tool calls from ConductResearch calls
            think_tool_calls = [
                tool_call for tool_call in most_recent_message.tool_calls if tool_call["name"] == "think_tool"
            ]

            conduct_research_calls = [
                tool_call for tool_call in most_recent_message.tool_calls if tool_call["name"] == "ConductResearch"
            ]

            refine_report_calls = [
                tool_call for tool_call in most_recent_message.tool_calls if tool_call["name"] == "refine_draft_report"
            ]

            # Handle think_tool calls (synchronous)
            for tool_call in think_tool_calls:
                observation = think_tool.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(content=observation, name=tool_call["name"], tool_call_id=tool_call["id"])
                )

            # Handle ConductResearch calls (asynchronous)
            if conduct_research_calls:
                # Launch parallel research agents
                coros = [
                    researcher_agent.ainvoke(
                        {
                            "researcher_messages": [HumanMessage(content=tool_call["args"]["research_topic"])],
                            "research_topic": tool_call["args"]["research_topic"],
                        }
                    )
                    for tool_call in conduct_research_calls
                ]
                tool_results = await asyncio.gather(*coros)

                # Format research results as tool messages
                research_tool_messages = [
                    ToolMessage(
                        content=result.get("compressed_research", "Error synthesizing research report"),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                    for result, tool_call in zip(tool_results, conduct_research_calls)
                ]
                tool_messages.extend(research_tool_messages)
                # Aggregate raw notes from all research
                all_raw_notes = ["\n".join(result.get("raw_notes", [])) for result in tool_results]
            for tool_call in refine_report_calls:
                notes = get_notes_from_tool_calls(supervisor_messages)
                findings = "\n".join(notes)

                draft_report = refine_draft_report.invoke(
                    {
                        "research_brief": state.get("research_brief", ""),
                        "findings": findings,
                        "draft_report": state.get("draft_report", ""),
                    }
                )

                tool_messages.append(
                    ToolMessage(content=draft_report, name=tool_call["name"], tool_call_id=tool_call["id"])
                )
        except Exception as e:
            should_end = True
            next_step = END

    # Single return point with appropriate state updates
    if should_end:
        return Command(
            goto=next_step,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", ""),
            },
        )
    elif len(refine_report_calls) > 0:
        return Command(
            goto=next_step,
            update={"supervisor_messages": tool_messages, "raw_notes": all_raw_notes, "draft_report": draft_report},
        )
    else:
        return Command(goto=next_step, update={"supervisor_messages": tool_messages, "raw_notes": all_raw_notes})


supervisor_builder = StateGraph(SupervisorState, context_schema=Configuration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)

supervisor_builder.add_edge(START, "supervisor")
supervisor_agent = supervisor_builder.compile(checkpointer=True)
