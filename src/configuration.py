"""Configuration management for the Open Deep Research system."""

import os
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""

    # LLM Configuration
    llm: BaseChatModel = Field(
        description="The language model to use for research operations.",
    )
    allow_clarification: bool = Field(
        default=True,
        description="Whether to allow the agent to ask clarifying questions to the user before starting research.",
    )
    context_window: int = Field(
        default=1048576,
        description="Context window (input) size of the LLM in tokens.",
    )

    researcher_tools: list[Any] | None = Field(
        default=None,
        description="List of tools available to the Researcher agent.",
    )
    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        description="Maximum number of retries for structured output calls from models",
    )
    max_concurrent_research_units: int = Field(
        default=5,
        description="Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits.",
    )
    max_researcher_iterations: int = Field(
        default=15,
        description="Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions.",
    )
    max_react_tool_calls: int = Field(
        default=10,
        description="Maximum number of tool calls for the Researcher agent in a single iteration.",
    )

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None = None) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
            if field_name != "llm"  # Skip LLM field for environment variable lookup
        }
        # Handle LLM separately - it comes from configurable, not environment
        if "llm" in configurable:
            values["llm"] = configurable["llm"]

        return cls(**{k: v for k, v in values.items() if v is not None})

    def get_model(self) -> BaseChatModel:
        """Get the configured LLM model or default fallback."""
        return self.llm

    def model_with_tool(self):
        """Get the LLM model with tools if configured.
        """
        if self.researcher_tools:
            tools = self.get_research_tools()
            return self.llm.bind_tools(tools)
        return self.llm

    def writer_model(self):
        """Get the LLM model for writing tasks.
        Model have to be more precise.
        """
        return self.llm

    def creative_model(self):
        """Get the LLM model for creative tasks.
        Model have to be more creative.
        Optionally, you can configure a different model here.
        """
        return self.llm

    def get_research_tools(self) -> list[Any]:
        """Get the configured tools for the Researcher agent."""
        return self.researcher_tools if self.researcher_tools else []

    def get_tool_instructions(self) -> str:
        """Get the tool instructions for the Researcher agent."""
        if not self.researcher_tools:
            return ""
        tool_instructions = "\n".join(
            [f"- **{tool.name}**: {tool.description}" for tool in self.researcher_tools if tool.name != "think_tool"]
        )
        return tool_instructions
