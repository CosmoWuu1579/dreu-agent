"""
LangGraph agent that designs a quantum feature map (Knipfer-style single-agent
tool-calling loop). The LLM holds a running message history and repeatedly calls
one tool -- evaluate_feature_map_tool -- refining its circuit from the returned
metrics until accuracy plateaus.

Graph (a ReAct loop with a reviewer):
    agent --(tool_call?)--> tools --(not done?)--> review --> agent ... --> END
The `review` node is a separate reviewer persona (Sakka et al.'s "Review"
component) that critiques each evaluated candidate and injects one concrete
suggestion back into the conversation before the designer's next turn.

Run:
    export ANTHROPIC_API_KEY=...             # required
    export ANTHROPIC_MODEL=claude-opus-4-8   # a model id your key can access
    export AGENT_DIM=2                        # feature/qubit dimensionality
    export AGENT_KERNEL=fidelity             # 'fidelity' (search) or 'dqc1' (paper)
    export AGENT_TRAIN_SIZE=20               # training points (per class for ad_hoc)
    export AGENT_TEST_SIZE=5                 # test points (per class for ad_hoc)
    python agent_graph.py

Every run writes a full transcript (messages, tool calls, results, best-so-far)
to runs/agent_run_<timestamp>.txt.
"""

from __future__ import annotations

import os
import json
import time
import datetime
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()
from eval_harness import (evaluate_feature_map, load_dataset,
                          SEED_FEATURE_MAP, seed_library_text)

# NOTE: set ANTHROPIC_MODEL to a model id your API key can access.
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MAX_ITERS = int(os.environ.get("AGENT_MAX_ITERS", "8"))    # max tool rounds
DIM = int(os.environ.get("AGENT_DIM", "2"))                # feature/qubit count
KERNEL = os.environ.get("AGENT_KERNEL", "fidelity")        # 'fidelity' or 'dqc1'
TRAIN_SIZE = int(os.environ.get("AGENT_TRAIN_SIZE", "20"))  # train points (per class for ad_hoc)
TEST_SIZE = int(os.environ.get("AGENT_TEST_SIZE", "5"))     # test points (per class for ad_hoc)
TEMPERATURE = float(os.environ.get("AGENT_TEMPERATURE", "0.1"))
PLOT = os.environ.get("AGENT_PLOT", "1") != "0"            # save a figure per round
TARGET_ACC = 1.0

# Load the benchmark once and reuse it for every candidate evaluation.
DATA = load_dataset(DIM, train_size=TRAIN_SIZE, test_size=TEST_SIZE)


# ---------------------------------------------------------------------------
# Logging: append a readable transcript of the whole run to a txt file
# ---------------------------------------------------------------------------
os.makedirs("runs", exist_ok=True)
LOG_PATH = os.path.join("runs", f"agent_run_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt")


def log(section: str, text: str) -> None:
    line = f"\n{'=' * 72}\n{section}\n{'=' * 72}\n{text}\n"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)


def _render(message) -> str:
    """Flatten an AI message (text blocks + tool calls) for the log."""
    parts = []
    content = message.content
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
    for call in getattr(message, "tool_calls", None) or []:
        parts.append(f"[tool_call {call['name']}] code:\n{call['args'].get('code', '')}")
    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# The single tool the agent can call
# ---------------------------------------------------------------------------
@tool
def evaluate_feature_map_tool(code: str) -> dict:
    """Train and score a candidate quantum feature map, returning its metrics.

    Args:
        code: Python defining `build_circuit(x) -> QuantumCircuit`. `x` is a
            numpy feature vector; `np` and `QuantumCircuit` are already in scope
            (no imports). The circuit must have no measurements and no trainable
            parameters.

    Returns:
        dict with `accuracy` (higher is better), `n_qubits`, `n_gates`, `depth`,
        or `{ok: False, error: ...}` if the code failed to run.
    """
    return evaluate_feature_map(code, data=DATA, kernel=KERNEL)


TOOLS = [evaluate_feature_map_tool]

SYSTEM = f"""You are an expert quantum circuit designer. Design a quantum feature
map for a {DIM}-feature binary classification task that MAXIMIZES SVM test
accuracy while keeping the gate count modest.

You have one tool: evaluate_feature_map_tool(code). Work iteratively --
1. Propose a feature map and call the tool.
2. Read the returned accuracy / n_gates / depth (or error) and refine.
3. Explore ideas: entanglement patterns, data re-uploading, different encoding
   rotations. Keep improving until accuracy plateaus near {TARGET_ACC}.

Feature-map contract (the `code` argument):
- Define exactly one function build_circuit(x) -> QuantumCircuit.
- x is a length-{DIM} numpy array. Use at least {DIM} qubits and at most {DIM + 4}.
- NO imports; `np` and `QuantumCircuit` are in scope. No measurements, no
  trainable parameters, no classical nonlinear preprocessing of x.

Call the tool on every turn until you are confident, then give a short final
summary WITHOUT a tool call."""


# ---------------------------------------------------------------------------
# Graph state: a running message list + bookkeeping
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration: int
    best: dict
    last_code: str          # code from the most recent evaluation
    last_metrics: dict       # metrics from the most recent evaluation


llm = ChatAnthropic(model=MODEL, temperature=TEMPERATURE, max_tokens=4096).bind_tools(TOOLS)
review_llm = ChatAnthropic(model=MODEL, temperature=0.3, max_tokens=1024)  # no tools

REVIEWER_SYSTEM = """You are a critical reviewer of quantum feature-map designs \
for a quantum-kernel SVM. Given the latest candidate circuit and its measured \
accuracy / gate stats, give ONE specific, actionable suggestion to improve test \
accuracy without unnecessary gate bloat -- e.g. change the entanglement pattern, \
add data re-uploading, adjust the encoding rotations, or reduce depth. Two \
sentences maximum. If the candidate errored, state the concrete fix instead."""


def agent_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    resp = llm.invoke([SystemMessage(content=SYSTEM)] + state["messages"])
    dt = time.perf_counter() - t0
    log(f"AGENT (after {state['iteration']} tool rounds) [llm {dt:.2f}s]", _render(resp))
    return {"messages": [resp]}


def tool_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    tool_messages, best = [], dict(state["best"])
    for call in last.tool_calls:
        code = call["args"].get("code", "")
        t0 = time.perf_counter()
        result = evaluate_feature_map(code, data=DATA, kernel=KERNEL,
                                      plot=PLOT, iteration=state["iteration"] + 1)
        dt = time.perf_counter() - t0
        result["eval_seconds"] = round(dt, 3)   # let the agent see runtime too
        log(f"TOOL RESULT (round {state['iteration'] + 1}) [eval {dt:.2f}s]",
            f"code:\n{code}\n\nresult: {json.dumps(result)}")
        if result.get("ok") and result["accuracy"] > best.get("accuracy", -1.0):
            best = {"code": code, "accuracy": result["accuracy"], "metrics": result}
        tool_messages.append(ToolMessage(content=json.dumps(result), tool_call_id=call["id"]))
    print(f"[round {state['iteration'] + 1}] best accuracy so far: {best.get('accuracy')}")
    return {"messages": tool_messages, "iteration": state["iteration"] + 1,
            "best": best, "last_code": code, "last_metrics": result}


def review_node(state: AgentState) -> dict:
    """Reviewer persona: critique the latest candidate and suggest one change."""
    code = state.get("last_code", "")
    m = state.get("last_metrics", {})
    if m.get("ok"):
        prompt = (f"Candidate stats: accuracy={m['accuracy']}, n_gates={m['n_gates']}, "
                  f"depth={m['depth']}, n_qubits={m['n_qubits']}.\n"
                  f"Code:\n```python\n{code}\n```\n"
                  f"Give one concrete improvement to raise test accuracy.")
    else:
        prompt = (f"The candidate errored: {m.get('error')}\n"
                  f"Code:\n```python\n{code}\n```\nGive the concrete fix in one sentence.")
    t0 = time.perf_counter()
    resp = review_llm.invoke([SystemMessage(content=REVIEWER_SYSTEM),
                              HumanMessage(content=prompt)])
    dt = time.perf_counter() - t0
    note = resp.content if isinstance(resp.content, str) else _render(resp)
    log(f"REVIEW (round {state['iteration']}) [llm {dt:.2f}s]", note)
    return {"messages": [HumanMessage(content=f"Reviewer feedback: {note}")]}


def route_after_agent(state: AgentState) -> str:
    if getattr(state["messages"][-1], "tool_calls", None):
        return "tools"            # agent proposed a candidate to evaluate
    return "stop"                 # agent gave a final answer, no tool call


def route_after_tools(state: AgentState) -> str:
    if state["iteration"] >= MAX_ITERS:                              # budget exhausted
        return "stop"
    if state["best"].get("accuracy", -1.0) >= TARGET_ACC - 0.01:    # target reached
        return "stop"
    return "review"               # otherwise get reviewer guidance, then loop


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_node("review", review_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "stop": END})
    g.add_conditional_edges("tools", route_after_tools, {"review": "review", "stop": END})
    g.add_edge("review", "agent")
    return g.compile()


if __name__ == "__main__":
    log("RUN CONFIG",
        f"model={MODEL} dim={DIM} kernel={KERNEL} max_iters={MAX_ITERS} "
        f"temperature={TEMPERATURE}\n"
        f"train_size={TRAIN_SIZE} test_size={TEST_SIZE} "
        f"(actual train/test points: {len(DATA[0])}/{len(DATA[2])})")

    first_message = HumanMessage(content=(
        "Design and iteratively improve the feature map. Here is a library of "
        "seed feature maps you may start from, adapt, or combine:\n\n"
        f"{seed_library_text()}\n\n"
        "Pick or adapt one, then call evaluate_feature_map_tool with your first "
        "candidate. Try to beat the best of these baselines."
    ))

    run_t0 = time.perf_counter()
    app = build_graph()
    final = app.invoke(
        {"messages": [first_message], "iteration": 0, "best": {"accuracy": -1.0}},
        {"recursion_limit": 100},
    )
    log("TOTAL RUNTIME", f"{time.perf_counter() - run_t0:.2f}s over {final['iteration']} rounds")

    best = final["best"]
    log("BEST RESULT", json.dumps(best, indent=2))

    print("\n" + "=" * 72)
    if best.get("code"):
        print("BEST FEATURE MAP")
        print("=" * 72)
        print(best["code"].strip())
        print("-" * 72)
        print("stats:", json.dumps(best.get("metrics", {"accuracy": best["accuracy"]}), indent=2))
    else:
        print("No valid feature map was produced (every candidate errored).")
    print("=" * 72)
    print("Transcript written to:", LOG_PATH)
