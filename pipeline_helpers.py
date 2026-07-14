"""Helpers for AgentPipeline, split out to keep that file focused on the
graph nodes and routing. Functionality is identical: message utilities,
the `from_model` factory, the per-pipeline tool factories (they close over
the pipeline instance), and small logging/notes utilities.
"""

from __future__ import annotations

import os
import re
import json
import time
from typing import Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from DiscoveryTask import DiscoveryTask
from PipelineState import PipelineState


def _extract_code(text) -> str:
    """Pull the LAST ```python fenced code block out of an LLM message
    ("" if none).

    Last, not first: a submission reply often recaps earlier attempts before
    presenting the final design, and the final design comes last. No raw-text
    fallback: prose must never be mistaken for a submission."""
    if isinstance(text, list):  # structured content blocks (Anthropic, etc.)
        text = "\n".join(b["text"] for b in text
                         if isinstance(b, dict) and b.get("type") == "text")
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text or "", re.DOTALL)
    return blocks[-1].strip() if blocks else ""


def _text_of(message) -> str:
    c = message.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(b["text"] for b in c
                         if isinstance(b, dict) and b.get("type") == "text")
    return str(c)


def _has_tool_calls(message) -> bool:
    return bool(getattr(message, "tool_calls", None))


# ---------------------------------------------------------------- factory
def from_model(
    cls,
    task: DiscoveryTask,
    model: str,
    *,
    model_provider: Optional[str] = None,
    generate_kwargs: Optional[dict] = None,
    explorer_kwargs: Optional[dict] = None,
    review_kwargs: Optional[dict] = None,
    deploy_kwargs: Optional[dict] = None,
    translate_kwargs: Optional[dict] = None,
    **pipeline_kwargs,
):
    """Build a pipeline from a single provider-agnostic model id.

    `model`/`model_provider` are passed to LangChain's ``init_chat_model``, so
    any supported provider works ("claude-*", "gpt-*", "gemini-*", ...). Each
    node's LLM is created from the same id with its own sampling kwargs
    (temperature, max_tokens, ...); pass ``review_kwargs=None`` etc. to reuse
    the generate model for that role. Remaining kwargs go to ``__init__``.

    (Exposed as the ``AgentPipeline.from_model`` classmethod.)
    """
    from langchain.chat_models import init_chat_model

    def _llm(kwargs: Optional[dict]):
        if kwargs is None:
            return None
        return init_chat_model(model, model_provider=model_provider, **kwargs)

    generate_llm = _llm(generate_kwargs or {"temperature": 0.4, "max_tokens": 4096})
    return cls(
        task,
        generate_llm=generate_llm,
        explorer_llm=_llm(explorer_kwargs),
        review_llm=_llm(review_kwargs),
        deploy_llm=_llm(deploy_kwargs),
        translate_llm=_llm(translate_kwargs),
        **pipeline_kwargs,
    )


# ------------------------------------------------------------------ tools
def make_evaluate_tool(pipeline):
    """Default evaluate tool: runs task.evaluate and returns metrics as JSON.

    Every call is stamped with a global `trial` number (also forwarded as
    `iteration=` to the evaluator, which e.g. tags plot filenames), so
    every artifact of the run traces back to the trial that produced it."""
    task = pipeline.task

    @tool
    def evaluate(code: str) -> str:
        """Run one TRIAL evaluation of a candidate and return metrics as JSON.

        `code` MUST be the full Python source that defines the entry-point
        function. Call this one candidate at a time: each result comes back
        to you, so analyze it before deciding your next trial. You have a
        limited budget of trials per design session; when it is spent you
        will be asked to analyze your results and present your final
        design. You can also stop early: when you are happy with a design,
        reply WITHOUT calling tools and it is submitted for official
        evaluation. Returns a JSON object with an "ok" flag, a "trial"
        number identifying this candidate, plus metrics, or an "error"
        string.
        """
        pipeline._eval_count += 1
        kwargs = dict(pipeline.evaluate_kwargs)
        kwargs.setdefault("iteration", pipeline._eval_count)
        result = task.evaluate(code, **kwargs)
        if isinstance(result, dict):
            result.setdefault("trial", pipeline._eval_count)
        return json.dumps(result)

    return evaluate


def make_check_notes_tool(pipeline):
    """Tool letting the designer re-read the explorer's distilled notes."""

    @tool
    def check_explorer_notes() -> str:
        """Return the explorer's distilled research notes for this run.

        Call this to re-read the background research (documentation
        findings, prior work, design advice) the explorer summarized
        before the design phase.
        """
        return (pipeline._explorer_notes.strip()
                or "(no explorer notes recorded yet)")

    return check_explorer_notes


def seed_block(task: DiscoveryTask, seed_type: Optional[str] = None) -> str:
    """The task's seed artifacts as markdown. With no seed_type, EVERY
    group in the library is included, one section per type."""
    types = [seed_type] if seed_type else (task.seed_types() or [None])
    parts = []
    for st in types:
        group = task.seeds(st)
        if not group:
            continue
        entries = "\n\n".join(f"### {n}\n```python\n{c.strip()}\n```"
                              for n, c in group.items())
        parts.append(f"## Seed type: {st}\n{entries}" if st else entries)
    return "\n\n".join(parts) or "(no seeds available)"


def make_seed_library_tool(pipeline):
    """Tool exposing the task's seed designs (replaces pasting them inline)."""

    @tool
    def view_seed_library(seed_type: str = "") -> str:
        """Return the library of known-good seed designs for this task.

        These are reference artifacts you may start from, adapt, or
        combine. By default the ENTIRE library is returned, grouped by
        seed type; pass `seed_type` to see just one group. Consult it
        before proposing your first candidate and whenever you want a
        proven pattern to build on.
        """
        return seed_block(pipeline.task, seed_type or None)

    return view_seed_library


def dedupe_tools(tools) -> list:
    seen: dict = {}
    for t in tools:
        seen[t.name] = t
    return list(seen.values())


def explorer_toolset(pipeline) -> list:
    """Everything the explorer can call: injected research tools (web /
    docs / papers -- optional) plus the seed library."""
    return dedupe_tools(pipeline.explorer_tools + [pipeline.seed_library_tool])


def design_tools(pipeline) -> list:
    """Everything the generate LLM can call: research tools, the seed
    library, the notes tool (only if use_notes_tool), and ALWAYS the
    evaluate tool."""
    tools = list(pipeline.generate_tools) + [pipeline.seed_library_tool]
    if pipeline.use_notes_tool:
        tools.append(pipeline.check_notes_tool)
    tools.append(pipeline.evaluate_tool)
    return dedupe_tools(tools)


# ------------------------------------------------------------------ misc
def sys_prompt(task: DiscoveryTask, agent_type: str, default: str) -> str:
    """Task system prompt for `agent_type`, falling back to `default`."""
    try:
        return task.system_prompt(agent_type)
    except KeyError:
        return default


def log_system(pipeline, role: str, content: str) -> None:
    """Log a role's system prompt on first use and whenever it changes
    (e.g. generate's gains the explorer notes after the handoff)."""
    if pipeline._logged_systems.get(role) != content:
        pipeline._logged_systems[role] = content
        pipeline.task.log(f"SYSTEM PROMPT ({role})", content)


def append_note(pipeline, section: str, text: str) -> None:
    """Append a finding to the persistent notes file (research storage)."""
    os.makedirs(os.path.dirname(pipeline.notes_path), exist_ok=True)
    with open(pipeline.notes_path, "a", encoding="utf-8") as f:
        f.write(f"\n## {section}\n{text}\n")


def invoke_tool(pipeline, tool_map: dict, call: dict,
                note: bool = False) -> tuple[ToolMessage, Optional[dict]]:
    """Run ONE tool call with timing and logging.

    `note=True` (explorer channel only) also appends the result to the
    notes file -- the notes file is the EXPLORER'S record, nothing else
    writes to it.

    Returns (ToolMessage for the graph state, parsed JSON dict or None).
    """
    tool_obj = tool_map.get(call["name"])
    t0 = time.perf_counter()
    try:
        out = (tool_obj.invoke(call["args"]) if tool_obj
               else json.dumps({"ok": False, "error": f"unknown tool {call['name']}"}))
    except Exception as exc:
        out = json.dumps({"ok": False, "error": f"tool raised: {exc}"})
    dt = time.perf_counter() - t0
    out = str(out)
    # log the INPUTS too (e.g. the exact code a trial evaluated), in full --
    # the transcript is the record of what was actually tested
    args_lines = []
    for k, v in (call.get("args") or {}).items():
        v_str = v if isinstance(v, str) else json.dumps(v)
        args_lines.append(f"{k}:\n{v_str}" if "\n" in v_str else f"{k}: {v_str}")
    args_view = "\n".join(args_lines) or "(no args)"
    pipeline.task.log(f"TOOL {call['name']} [tool {dt:.2f}s]",
                      f"--- args ---\n{args_view}\n--- result ---\n{out[:]}")
    if note:
        append_note(pipeline, f"TOOL {call['name']}", out[:])
    try:
        parsed = json.loads(out)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    return (ToolMessage(content=out, tool_call_id=call["id"], name=call["name"]),
            parsed)


def done(pipeline, state: PipelineState) -> bool:
    """Deterministic stop check: budget exhausted or target reached."""
    if state.get("iteration", 0) >= pipeline.max_iters:
        return True
    best = state.get("best") or {}
    return best.get(pipeline.target_metric, -1.0) >= pipeline.target_value - 1e-9


def best_so_far_note(pipeline, state: PipelineState) -> str:
    """Factual 'current best' line, injected into the DESIGN THREAD at
    session start (not the system prompt) so it sits after the variant it
    names -- correct chronological order -- and keeps the system prompt
    static. The how-to (build on it / when to resubmit) lives in the task's
    generate prompt; this only supplies the live numbers. Empty until a
    best exists (session 1, where the task prompt points at the best seed)."""
    best = state.get("best") or {}
    if not best.get("code"):
        return ""
    bm = best.get("metrics") or {}
    return (
        f"Current best is variant #{best.get('variant', '?')} "
        f"({pipeline.target_metric}={best.get(pipeline.target_metric)}, "
        f"gates={bm.get('n_gates')}, depth={bm.get('depth')}) -- it is saved "
        "and cannot be lost. Build on it per your Strategy (refine it to beat "
        "it, or try a fresh direction if it is weak).")
