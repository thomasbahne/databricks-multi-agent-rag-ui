"""
RAG Agent using ResponsesAgent with VectorSearchRetrieverTool
Deploy via agents.deploy() after registering in Unity Catalog
"""

import yaml
from pathlib import Path

import mlflow
from databricks_langchain import (
    ChatDatabricks,
    VectorSearchRetrieverTool,
)
from langgraph.prebuilt import create_react_agent

mlflow.langchain.autolog()


def load_config(config_path: str) -> dict:
    """Load agent configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_rag_agent(config: dict):
    """
    Create a RAG agent with vector search retrieval.

    Args:
        config: Dict with keys:
            - agent_name: Display name
            - vector_search_endpoint: VS endpoint name
            - vector_search_index: Full index name (catalog.schema.index)
            - llm_endpoint: Model serving endpoint for LLM
            - system_prompt: Agent instructions
    """
    # Configure retriever tool
    retriever_tool = VectorSearchRetrieverTool(
        index_name=config["vector_search_index"],
        num_results=5,
        columns=["content", "source", "chunk_id"],
        filters={},
        text_column="content",
        tool_name="search_documents",
        tool_description=(
            f"Search the {config['agent_name']} knowledge base. "
            "Use this tool to find relevant information for answering questions."
        ),
    )

    # Configure LLM
    llm = ChatDatabricks(
        endpoint=config.get("llm_endpoint", "databricks-meta-llama-3-3-70b-instruct"),
        temperature=0.1,
        max_tokens=1024,
    )

    # Create ReAct agent with retriever tool
    agent = create_react_agent(
        model=llm,
        tools=[retriever_tool],
        state_modifier=config.get(
            "system_prompt",
            "You are a helpful assistant. Use the search tool to find relevant "
            "information before answering questions. Always cite your sources.",
        ),
    )

    return agent


def get_agent(config_name: str = "config_agent_a.yml"):
    """
    Entry point for MLflow model serving.
    Loads config relative to this file and returns agent.
    """
    config_path = Path(__file__).parent / config_name
    config = load_config(str(config_path))
    return create_rag_agent(config)


# MLflow model signature for logging
mlflow.models.set_model(get_agent())
