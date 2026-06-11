"""
workflows/langgraph_workflow.py

LangGraph Workflow for the Agentic Question Paper Generator.

This module constructs and compiles the StateGraph that orchestrates
all agent nodes in sequence with conditional error routing.

Workflow:
    START
      → syllabus_agent
      → question_generator_agent
      → bloom_agent
      → validation_agent
      → answerkey_agent
      → END

Error Routing:
    After each agent, if state["status"] == "failed",
    the workflow routes directly to END to stop further execution.

Usage:
    from app.workflows.langgraph_workflow import create_workflow, build_initial_state

    compiled = create_workflow()
    initial  = build_initial_state(rag_chunks=..., syllabus_context=..., ...)
    result   = compiled.invoke(initial)
"""

from typing import Any

from langgraph.graph import END, StateGraph

from app.agents.answerkey_agent import answerkey_agent_node
from app.agents.bloom_agent import bloom_agent_node
from app.agents.question_generator_agent import question_generator_agent_node
from app.agents.syllabus_agent import syllabus_agent_node
from app.agents.validation_agent import validation_agent_node
from app.models.state import AgentState, PaperMetadata, QuestionDistribution, RagChunk
from app.services.logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Node name constants — single source of truth for graph wiring
# ---------------------------------------------------------------------------
NODE_SYLLABUS = "syllabus_agent"
NODE_QUESTION = "question_generator_agent"
NODE_BLOOM = "bloom_agent"
NODE_VALIDATION = "validation_agent"
NODE_ANSWERKEY = "answerkey_agent"


# ---------------------------------------------------------------------------
# Conditional routing function
# ---------------------------------------------------------------------------

def _route_after_agent(state: AgentState) -> str:
    """
    Routing function used after every agent node.

    Returns:
        "failed": route to END immediately if the agent failed.
        "ok":     continue to the next agent node.
    """
    if state.get("status") == "failed":
        logger.warning(
            f"Workflow routing: status is 'failed' after "
            f"'{state.get('current_agent', 'unknown')}'. Terminating early."
        )
        return "failed"
    return "ok"


# ---------------------------------------------------------------------------
# Workflow factory
# ---------------------------------------------------------------------------

def create_workflow() -> Any:
    """
    Build and compile the LangGraph StateGraph.

    Nodes are wired in sequence with conditional edges so that a
    failure in any node halts the workflow immediately.

    Returns:
        A compiled LangGraph CompiledStateGraph ready to invoke.
    """
    logger.info("Building LangGraph workflow...")

    graph = StateGraph(AgentState)

    # ---- Register nodes ----
    graph.add_node(NODE_SYLLABUS,   syllabus_agent_node)
    graph.add_node(NODE_QUESTION,   question_generator_agent_node)
    graph.add_node(NODE_BLOOM,      bloom_agent_node)
    graph.add_node(NODE_VALIDATION, validation_agent_node)
    graph.add_node(NODE_ANSWERKEY,  answerkey_agent_node)

    # ---- Entry point ----
    graph.set_entry_point(NODE_SYLLABUS)

    # ---- Conditional edges: continue or stop after each agent ----
    graph.add_conditional_edges(
        NODE_SYLLABUS,
        _route_after_agent,
        {"ok": NODE_QUESTION, "failed": END},
    )
    graph.add_conditional_edges(
        NODE_QUESTION,
        _route_after_agent,
        {"ok": NODE_BLOOM, "failed": END},
    )
    graph.add_conditional_edges(
        NODE_BLOOM,
        _route_after_agent,
        {"ok": NODE_VALIDATION, "failed": END},
    )
    graph.add_conditional_edges(
        NODE_VALIDATION,
        _route_after_agent,
        {"ok": NODE_ANSWERKEY, "failed": END},
    )

    # ---- Final node always routes to END ----
    graph.add_edge(NODE_ANSWERKEY, END)

    compiled = graph.compile()
    logger.info("LangGraph workflow compiled successfully.")
    return compiled


# ---------------------------------------------------------------------------
# Initial state builder
# ---------------------------------------------------------------------------

def build_initial_state(
    rag_chunks: list[RagChunk],
    syllabus_context: str,
    content_context: str,
    distribution: QuestionDistribution,
    paper_metadata: PaperMetadata | None = None,
) -> AgentState:
    """
    Construct a fully-initialised AgentState dict to pass to the workflow.

    All list and optional fields are set to their empty/None defaults so
    LangGraph's merge logic works correctly from the first node.

    Args:
        rag_chunks:         All document chunks from RAG ingestion.
        syllabus_context:   Formatted retrieved chunks for syllabus extraction.
        content_context:    Formatted retrieved chunks for question generation.
        distribution:       Question distribution parameters (marks, counts, difficulty).
        paper_metadata:     Optional PDF header metadata (institution, course, etc.).

    Returns:
        A complete AgentState dict ready to invoke the compiled workflow.
    """
    return AgentState(
        rag_chunks=rag_chunks,
        syllabus_context=syllabus_context,
        content_context=content_context,
        syllabus_topics=[],
        question_distribution=distribution,
        generated_questions=[],
        bloom_analysis=[],
        validated_questions=[],
        answer_key=[],
        paper_metadata=paper_metadata,
        final_pdf_path=None,
        answer_key_pdf_path=None,
        errors=[],
        current_agent=None,
        status="initialized",
    )
