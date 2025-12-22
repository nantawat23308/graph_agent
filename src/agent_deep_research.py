from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.config import RunnableConfig
from src.utility import get_today_str
from src.deep_research.state import AgentState, AgentInputState

from src.deep_research.agent_scope import write_draft_report, write_research_brief, clarify_with_user
from src.deep_research.agent_supervisor import supervisor_agent
from src.prompts.prompt_final import final_report_generation_with_helpfulness_insightfulness_hit_citation_prompt
from src.configuration import Configuration
from langgraph.checkpoint.memory import InMemorySaver


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """
    Final report generation node.

    Synthesizes all research findings into a comprehensive final report
    """
    configurable = Configuration.from_runnable_config(config)
    writer_model = configurable.writer_model()
    notes = state.get("notes", [])

    findings = "\n".join(notes)

    final_report_prompt = final_report_generation_with_helpfulness_insightfulness_hit_citation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str(),
        draft_report=state.get("draft_report", ""),
        user_request=state.get("user_request", ""),
    )

    final_report = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])

    return {
        "final_report": final_report.content,
        "messages": ["Here is the final report: " + final_report.content],
    }


# Build the overall workflow
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState, context_schema=Configuration)
# Add workflow nodes
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("write_draft_report", write_draft_report)
deep_researcher_builder.add_node("supervisor_subgraph", supervisor_agent)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

# Add workflow edges
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", "write_draft_report")
deep_researcher_builder.add_edge("write_draft_report", "supervisor_subgraph")
deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# The completed graph can now be used to run the deep research agent workflow
checkpointer = InMemorySaver()
deep_researcher_graph = deep_researcher_builder.compile(checkpointer=checkpointer)
