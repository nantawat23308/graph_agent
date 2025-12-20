from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from src.langchain_milvus import constant
from src.langchain_milvus.searching import search_retrieve

load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from typing import List, TypedDict
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from src.configuration import Configuration
from src.logger import logger
from src.prompts import prompt_rag
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
import uuid

MAX_RETRIES = 3
MAX_INVOKES = 15


class AgentState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    retry_count: int = 0
    invoke_count: int = 0

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


class GradeHallucination(BaseModel):
    """Binary score for hallucination check in generation."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")


retriever = search_retrieve(constant.COLLECTION_NAME, constant.URI)

def retrieve(state: AgentState, config: RunnableConfig):
    logger.debug("--- RETRIEVING ---")
    docs = retriever.invoke(state["question"])
    return {"documents": docs}


def generate(state: AgentState, config: RunnableConfig):
    logger.info("--- GENERATING ---")
    # Standard RAG generation logic
    configurable = Configuration.from_runnable_config(config)
    invoke_count = state.get("invoke_count", 0) + 1
    if invoke_count > MAX_INVOKES:
        logger.info("--- MAX INVOKE COUNT REACHED: EXITING ---")
        return Command(
            goto=END,
            update={
                "generation": "Not able to generate a satisfactory answer within the allowed attempts.",
                "invoke_count": invoke_count}
        )

    llm = configurable.get_model()

    context = "\n".join([d.page_content for d in state["documents"]])
    human_template = ChatPromptTemplate.from_template(prompt_rag.HUMAN_CONTENT)
    prompt = human_template.format(context_text=context, user_question=state['question'])

    response = llm.invoke([SystemMessage(prompt_rag.SYSTEM_INSTRUCTION), HumanMessage(prompt)])
    logger.info("Invoke count: {}".format(invoke_count))
    return {"generation": response.content, "invoke_count": invoke_count}


def rewrite_query(state: AgentState, config: RunnableConfig):
    logger.debug("--- REWRITING QUERY ---")
    # Ask LLM to improve the question for better searching
    configurable = Configuration.from_runnable_config(config)
    llm = configurable.get_model()
    better_query = llm.invoke(f"Rewrite this question for a vector search: {state['question']}")
    return {"question": better_query.content, "retry_count": state.get("retry_count", 0) + 1}


def grade_documents(state: AgentState, config: RunnableConfig):
    logger.debug("--- CHECKING DOCUMENT RELEVANCE ---")
    question = state["question"]
    docs = state["documents"]
    if state.get("retry_count", 0) >= MAX_RETRIES:
        logger.info("--- MAX RETRIES REACHED: EXITING ---")
        return "final_fallback"
    # If Milvus returned nothing, handle it immediately
    if not docs:
        return "rewrite"
    configurable = Configuration.from_runnable_config(config)
    llm = configurable.get_model()
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Score the docs
    score = structured_llm_grader.invoke(f"Question: {question} \n Document: {docs[0].page_content}")
    if score.binary_score == "yes":
        return "generate"
    else:
        return "rewrite"


def grade_generation_v_documents(state: AgentState, config: RunnableConfig):
    logger.debug("--- SELF-CORRECTION: CHECKING HALLUCINATION ---")
    generation = state["generation"]
    docs = state["documents"]
    safety_result = check_safety(state)  # returns "stop" or "continue"
    if safety_result == "stop":
        return "final_fallback"
    configurable = Configuration.from_runnable_config(config)
    llm = configurable.get_model()
    hallucination_grader = llm.with_structured_output(GradeHallucination)
    score = hallucination_grader.invoke(f"Docs: {docs} \n Answer: {generation}")

    if score.binary_score == "yes":
        logger.debug("--- DECISION: ANSWER IS GROUNDED ---")
        return "useful"
    else:
        logger.debug("--- DECISION: HALLUCINATION DETECTED, RETRYING ---")
        return "not useful"


def generate_router(state: AgentState, config: RunnableConfig):
    """
    Single router to handle Safety and Hallucinations in order.
    """
    # 1. Priority: Check Safety
    safety_result = check_safety(state)  # returns "stop" or "continue"
    if safety_result == "stop":
        return "final_fallback"

    # 2. Check Hallucinations / Quality
    grading_result = grade_generation_v_documents(state, config)  # returns "useful" or "not useful"
    if grading_result == "not useful":
        return "generate"  # Loop back to fix hallucination

    # 3. If safe and useful, finish
    return "useful"


def check_safety(state: AgentState):
    if state["invoke_count"] > MAX_INVOKES:
        return "stop"
    return "continue"


def fallback_answer(state: AgentState):
    """A node to return a polite 'I don't know' instead of crashing or looping."""
    return {"generation": "I'm sorry, I searched my knowledge base multiple times but could not find a confident answer to your question."}


workflow = StateGraph(AgentState,
                      context_schema=Configuration)

# 1. Define Nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("rewrite", rewrite_query) # Rewrites question if docs are bad
workflow.add_node("final_fallback", fallback_answer) # The "giving up" node
# 2. Build Flow
workflow.set_entry_point("retrieve")

# Checkpoint 1: Are the docs relevant?
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        "generate": "generate",
        "rewrite": "rewrite", # Handling "No Results"
        "final_fallback": "final_fallback"
    }
)

# After rewriting, go back to retrieve
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)
workflow.add_edge("final_fallback", END)

# Checkpoint 2: Is the generation a hallucination? (Self-Correction)
workflow.add_conditional_edges(
    "generate",
    generate_router,
    {
        "final_fallback": "final_fallback",
        "generate": "generate",
        "useful": END
    }
)
checkpointer = InMemorySaver()
agent_app = workflow.compile(checkpointer)
# image = agent_app.get_graph(xray=True).draw_mermaid()
# with open('deep_researcher_graph.mmd', 'w') as f:
#     f.write(image)



if __name__ == '__main__':
    pass
    thread_id = uuid.uuid4()
    model = init_chat_model("bedrock_converse:us.meta.llama4-maverick-17b-instruct-v1:0")
    config: RunnableConfig = {
        "configurable":
            {
                "llm": model,
                "allow_clarification": False,
                "thread_id": thread_id
            },
        "recursion_limit": 100}
    inputs = {"question": "What are some exciting projects that can be built with LangChain?"}
    output = agent_app.invoke(
        inputs,
        config = config,
    )
    # for output in agent_app.stream(inputs):
    #     for key, value in output.items():
    #         logger.info(f"Node '{key}' completed.")

    # Final answer
    logger.info("Final Answer:")
    logger.info(output.get("generation"))