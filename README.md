



# Deep Research Agent

A deep research agent that can clarify user requirements, write a research brief, draft a report, and generate a final report after supervisor review.

## Project Overview

The Deep Research Agent is a sophisticated AI-powered agent designed to perform in-depth research on a given topic. It follows a structured workflow to ensure the final report is comprehensive, accurate, and tailored to the user's needs. The agent is built using the LangChain framework and leverages a multi-layered agentic architecture to manage the research process.

## Architecture

The agent's architecture is composed of several specialized agents that collaborate to produce the final research report. The workflow is as follows:

1.  **Clarify with User**: The agent starts by analyzing the user's request to determine if any clarification is needed. If the scope is unclear, it will ask clarifying questions.
2.  **Write Research Brief**: Once the scope is clear, the agent generates a detailed research brief that will guide the research process.
3.  **Supervisor Agent**: A supervisor agent breaks down the research brief into smaller, manageable tasks and delegates them to researcher agents.
4.  **Researcher Agent**: These agents conduct the actual research using a variety of tools, such as web search and a Milvus vector database.
5.  **Draft Report**: The findings from the researcher agents are used to generate a draft report.
6.  **Final Report Generation**: The draft report is then refined and formatted to produce the final, comprehensive report.

### Flowchart

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	clarify_with_user(clarify_with_user)
	write_research_brief(write_research_brief)
	write_draft_report(write_draft_report)
	final_report_generation(final_report_generation)
	__end__([<p>__end__</p>]):::last
	__start__ --> clarify_with_user;
	clarify_with_user -.-> __end__;
	clarify_with_user -.-> write_research_brief;
	supervisor_subgraph\3a__end__ --> final_report_generation;
	write_draft_report -.-> __end__;
	write_draft_report --> supervisor_subgraph\3a__start__;
	write_research_brief -.-> write_draft_report;
	final_report_generation --> __end__;
	subgraph supervisor_subgraph
	supervisor_subgraph\3a__start__(<p>__start__</p>)
	supervisor_subgraph\3asupervisor(supervisor)
	supervisor_subgraph\3a__end__(<p>__end__</p>)
	supervisor_subgraph\3a__start__ --> supervisor_subgraph\3asupervisor;
	supervisor_subgraph\3asupervisor -.-> supervisor_subgraph\3asupervisor_tools\3a__start__;
	supervisor_subgraph\3asupervisor_tools\3acompress_research -.-> supervisor_subgraph\3a__end__;
	supervisor_subgraph\3asupervisor_tools\3acompress_research -.-> supervisor_subgraph\3asupervisor;
	subgraph supervisor_tools
	supervisor_subgraph\3asupervisor_tools\3a__start__(<p>__start__</p>)
	supervisor_subgraph\3asupervisor_tools\3allm_call(llm_call)
	supervisor_subgraph\3asupervisor_tools\3atool_node(tool_node)
	supervisor_subgraph\3asupervisor_tools\3acompress_research(compress_research)
	supervisor_subgraph\3asupervisor_tools\3a__start__ --> supervisor_subgraph\3asupervisor_tools\3allm_call;
	supervisor_subgraph\3asupervisor_tools\3allm_call -.-> supervisor_subgraph\3asupervisor_tools\3acompress_research;
	supervisor_subgraph\3asupervisor_tools\3allm_call -.-> supervisor_subgraph\3asupervisor_tools\3atool_node;
	supervisor_subgraph\3asupervisor_tools\3atool_node --> supervisor_subgraph\3asupervisor_tools\3allm_call;
	end
	end
	classDef default fill:#f2f0f,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6f

```

## File-by-File Breakdown

### `main.py`

The entry point of the application. It initializes the language model, tools, and the main agent graph. It also contains the main asynchronous function to run the agent.

### `src/`

This directory contains the core logic for the agent.

-   **`agent_top.py`**: Defines the main graph that orchestrates the entire research process, connecting all the different agents and nodes.
-   **`agent_scope.py`**: Contains the logic for the agent that clarifies the user's request and writes the research brief.
-   **`agent_supervisor.py`**: Implements the supervisor agent that breaks down the research brief and delegates tasks to researcher agents.
-   **`agent_research.py`**: Defines the researcher agent that performs the actual research using the provided tools.
-   **`configuration.py`**: Contains the configuration classes for the agent, including LLM settings, tool configurations, and other parameters.
-   **`state.py`**: Defines the state objects that are passed between the different nodes in the agent graph.
-   **`utility.py`**: Provides utility functions used across the project, such as getting the current date, and tools like `think_tool`.

### `src/prompts/`

This directory contains the prompt templates used by the different agents.

-   **`prompt_final.py`**: Prompt for generating the final report.
-   **`prompt_research.py`**: Prompts for the researcher agent.
-   **`prompt_scope.py`**: Prompts for the scoping agent.
-   **`prompt_supervisor.py`**: Prompt for the supervisor agent.
-   **`prompt_utility.py`**: Prompts for utility functions.

### `src/langchain_milvus/`

This directory contains the integration with the Milvus vector database.

-   **`constant.py`**: Defines constants for the Milvus connection.
-   **`db.py`**: Contains functions for creating and managing Milvus vector store instances.
-   **`chunking.py`**: Provides functions for chunking text files.
-   **`ingest_data.py`**: Includes functions for ingesting data into the Milvus vector store.
-   **`process.py`**: A script that processes and ingests a file into the Milvus database.
-   **`searching.py`**: Implements various search functionalities.
-   **`rag.py`**: Contains the implementation of the RAG pipeline.
-   **`tools_utils.py`**: Defines tools for performing vector store searches.
-   **`utility.py`**: Provides utility functions for the Milvus integration.
-   **`demo.py`**: A demonstration script.
-   **`server.py`**: A script for managing the Milvus database.

## Setup and Usage

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Set up Environment Variables**:
    Create a `.env` file in the root directory and add the necessary environment variables (e.g., `TAVILY_SEARCH_API_KEY`, AWS credentials for Bedrock).
3.  **Run the Agent**:
    Execute the `main.py` script to run the agent.
    ```bash
    python main.py
    ```

## Tools

The agent uses the following tools to perform research:

-   **`tavily_search`**: A tool for performing web searches using the Tavily API.
-   **`think_tool`**: A tool for strategic reflection and planning during the research process.
-   **`milvus_search`**: A tool for searching the Milvus vector store for relevant documents.
-   **`rag_milvus`**: A tool that performs RAG search using the Milvus vector store and a language model.

