import asyncio
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from src.utility import get_today_str
from src.configuration import Configuration
from src.prompts.prompt_scope import (
    clarify_with_user_instructions,
    transform_messages_into_research_topic_human_msg_prompt,
    draft_report_generation_prompt,
)
from src.state import ClarifyWithUser, AgentState, ResearchQuestion, DraftReport


async def clarify_with_user(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.

    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.

    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences

    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        return Command(goto="write_research_brief")

    # Prepare the model for structured clarification analysis
    messages = state["messages"]

    # Configure model with structured output and retry logic
    clarification_model = (
        configurable.get_model()
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    # Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(messages=get_buffer_string(messages), date=get_today_str())
    response = None
    while not response:
        response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
        print("clarification response:", response)

    # Route based on clarification analysis
    if response.need_clarification:
        # End with clarifying question for user
        return Command(goto=END, update={"messages": [AIMessage(content=response.question)]})
    else:
        # Proceed to research with verification message
        return Command(goto="write_research_brief", update={"messages": [AIMessage(content=response.verification)]})


def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["write_draft_report"]]:
    """
    Transform the conversation history into a comprehensive research brief.

    Uses structured output to ensure the brief follows the required format
    and contains all necessary details for effective research.
    """
    configurable = Configuration.from_runnable_config(config)
    model = configurable.get_model()
    # Set up structured output model
    structured_output_model = model.with_structured_output(ResearchQuestion)

    # Generate research brief from conversation history
    response = structured_output_model.invoke(
        [
            HumanMessage(
                content=transform_messages_into_research_topic_human_msg_prompt.format(
                    messages=get_buffer_string(state.get("messages", [])), date=get_today_str()
                )
            )
        ]
    )

    # Update state with generated research brief and pass it to the supervisor
    return Command(goto="write_draft_report", update={"research_brief": response.research_brief})


def write_draft_report(state: AgentState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    """
    Final report generation node.

    Synthesizes all research findings into a comprehensive final report
    """
    configurable = Configuration.from_runnable_config(config)
    creative_model = configurable.creative_model()
    # Set up structured output model
    structured_output_model = creative_model.with_structured_output(DraftReport)
    research_brief = state.get("research_brief", "")
    draft_report_prompt = draft_report_generation_prompt.format(research_brief=research_brief, date=get_today_str())

    response = structured_output_model.invoke([HumanMessage(content=draft_report_prompt)])

    return {
        "research_brief": research_brief,
        "draft_report": response.draft_report,
        "supervisor_messages": ["Here is the draft report: " + response.draft_report, research_brief],
    }
