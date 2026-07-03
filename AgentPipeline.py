r"""
AgentPipeline: a reusable LangGraph pipeline for LLM-driven algorithm discovery.

It is generic over a DiscoveryTask (which supplies the domain: prompts, seeds,
evaluator, knowledge). The graph is fixed; only the task and the per-node tools
change.

THREE message channels keep the agents separate:

  explore_messages : the explorer's own research thread (tool calls + results).
  messages         : the design thread (generate + its tools + handoffs).
  review_messages  : the reviewer's own thread (it remembers its past advice).

Raw threads never leak across channels -- only distilled strings cross over
(explorer `notes`, evaluation batch summaries, reviewer feedback). So each
agent sees the others' conclusions, not their plumbing.

   START -> [explorer <--> explore_tools]        (optional: use_explorer=False
              |  (hand off notes)                 removes it from the graph)
              v
           generate <--> gen_tools --(research call)--> generate
              ^  ^   |         |
              |  |   |         | (evaluate call(s), <= max_variants_per_turn)
              |  |   |         v
              |  |   |      evaluate     deterministic: fold batch into best
              |  |   |         |
              |  |   |  (not done) | (done)
              |  +---|-- review <--+   |
              |      |     |           v
              +(feedback)--+        deploy -> END
                     |                 ^
                     +-----------------+   (plain reply after evaluating:
                                            "I'm satisfied" -> ship best-so-far)

  explorer : optional ENTRY. LLM with research tools (web search / docs /
             papers -- all injected, all optional) plus `view_seed_library`.
             Its written notes from EVERY round accumulate into `notes`, which
             is injected into the generate LLM's SYSTEM prompt (present from
             the first design call onward). Raw tool outputs go only to the
             notes file -- the explorer's private record -- which the designer
             can read via the `check_explorer_notes` tool.
  generate : LLM with research tools + `view_seed_library` +
             `check_explorer_notes` (iff the explorer exists) + the evaluate
             tool. It proposes circuits BY CALLING evaluate -- the tool-call
             args carry the code -- up to max_variants_per_turn variants per
             turn. A plain reply after it has evaluated at least once means
             "done": the graph deploys the best candidate found so far.
  gen_tools: custom tool executor for the design thread. Evaluate calls run
             in a for loop that STOPS at the first invalid (errored) variant;
             every tool_call id still receives a ToolMessage (skipped ones get
             an explicit marker) so the chat state stays valid.
  evaluate : DETERMINISTIC record step. The generate LLM already chose what
             code to run (via its tool calls); this node folds every result of
             the batch into best-so-far and summarizes it for the reviewer.
  review   : LLM critique with its own persistent thread; runs after every
             evaluated batch. Feedback is injected into the design thread.
  deploy   : final summary (+ optional user deploy_prompt + translate LLM).

Every LLM call is logged with its wall time ([llm X.XXs]); every individual
tool call likewise ([tool X.XXs]).
"""

from __future__ import annotations

import os
import re
import json
import time
from typing import Annotated, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)
from langchain_core.tools import tool

from DiscoveryTask import DiscoveryTask


class PipelineState(TypedDict, total=False):
    messages: Annotated[list, add_messages]          # design thread
    explore_messages: Annotated[list, add_messages]  # explorer's own thread
    review_messages: Annotated[list, add_messages]   # reviewer's own thread
    notes: str               # distilled explorer notes handed to the designer
    iteration: int           # variants evaluated so far
    current_code: str        # best candidate of the latest batch
    last_metrics: dict
    best: dict
    review: str
    explore_rounds: int      # explorer<->explore_tools loops (guard)
    gen_no_tool: int         # consecutive generate turns with no tool call (guard)


def _extract_code(text) -> str:
    """Pull a python code block out of an LLM message (fallback: raw text)."""
    if isinstance(text, list):  # structured content blocks (Anthropic, etc.)
        text = "\n".join(b["text"] for b in text
                         if isinstance(b, dict) and b.get("type") == "text")
    m = re.search(r"```(?:python)?\s*(.*?)```", text or "", re.DOTALL)
    return (m.group(1) if m else (text or "")).strip()


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


class AgentPipeline:
    def __init__(
        self,
        task: DiscoveryTask,
        generate_llm,
        *,
        use_explorer: bool = True,               # toggle the explorer subgraph
        explorer_llm=None,
        review_llm=None,
        deploy_llm=None,
        translate_llm=None,
        explorer_tools: Optional[list] = None,   # research tools for the explorer
        generate_tools: Optional[list] = None,   # research tools for generate
        evaluate_tool=None,                      # THE evaluate tool; default wraps task.evaluate
        review_tools: Optional[list] = None,
        deploy_prompt: Optional[str] = None,     # optional user instructions for deploy
        max_iters: int = 8,
        target_metric: str = "accuracy",
        target_value: float = 1.0,
        entry_point: str = "build_circuit",
        translate_spec: Optional[str] = None,
        evaluate_kwargs: Optional[dict] = None,
        max_explore_rounds: int = 4,
        max_gen_no_tool: int = 2,
        max_tool_rounds: int = 3,
        max_variants_per_turn: int = 3,          # evaluate calls allowed per generate turn
    ):
        self.task = task
        self.evaluate_kwargs = evaluate_kwargs or {}
        self.generate_llm = generate_llm
        self.explorer_llm = explorer_llm or generate_llm
        self.review_llm = review_llm or generate_llm
        self.deploy_llm = deploy_llm or generate_llm
        self.translate_llm = translate_llm
        self.use_explorer = use_explorer
        self.explorer_tools = explorer_tools or []
        self.generate_tools = generate_tools or []
        self.review_tools = review_tools or []
        # there is exactly ONE evaluate tool; it always exists (task fallback)
        self.evaluate_tool = evaluate_tool or self._make_evaluate_tool()
        self.evaluate_tool_name = self.evaluate_tool.name
        self.deploy_prompt = deploy_prompt
        self.max_iters = max_iters
        self.target_metric = target_metric
        self.target_value = target_value
        self.entry_point = entry_point
        self.translate_spec = translate_spec
        self.max_explore_rounds = max_explore_rounds
        self.max_gen_no_tool = max_gen_no_tool
        self.max_tool_rounds = max_tool_rounds
        self.max_variants_per_turn = max_variants_per_turn
        # a persistent side-file where explorer notes + tool findings accumulate
        self.notes_path = os.path.splitext(task.log_path)[0] + "_notes.md"
        # designer-side tool to re-read the notes file (only wired if explorer on)
        self.check_notes_tool = self._make_check_notes_tool()
        # seed reference designs, exposed as a tool (explorer + generate)
        self.seed_library_tool = self._make_seed_library_tool()

    # ---------------------------------------------------------------- factory
    @classmethod
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
    ) -> "AgentPipeline":
        """Build a pipeline from a single provider-agnostic model id.

        `model`/`model_provider` are passed to LangChain's ``init_chat_model``, so
        any supported provider works ("claude-*", "gpt-*", "gemini-*", ...). Each
        node's LLM is created from the same id with its own sampling kwargs
        (temperature, max_tokens, ...); pass ``review_kwargs=None`` etc. to reuse
        the generate model for that role. Remaining kwargs go to ``__init__``.
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
    def _make_evaluate_tool(self):
        """Default evaluate tool: runs task.evaluate and returns metrics as JSON."""
        task = self.task
        kwargs = self.evaluate_kwargs

        @tool
        def evaluate(code: str) -> str:
            """Run one candidate artifact end-to-end and return its metrics as JSON.

            `code` MUST be the full Python source that defines the entry-point
            function. Call this once per variant you want to test -- you may
            call it up to 3 times in a single turn to compare architectures.
            If a variant errors, later variants in the same turn are skipped,
            so put your most promising candidate first. Returns a JSON object
            with an "ok" flag plus metrics, or an "error" string.
            """
            return json.dumps(task.evaluate(code, **kwargs))

        return evaluate

    def _make_check_notes_tool(self):
        """Tool letting the designer re-read the explorer's notes file."""
        notes_path = self.notes_path

        @tool
        def check_explorer_notes() -> str:
            """Return the explorer's accumulated research notes for this run.

            Call this to re-read the background research (documentation
            findings, prior work, design advice) gathered before -- and
            during -- the design phase.
            """
            try:
                with open(notes_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            except FileNotFoundError:
                text = ""
            return text or "(no explorer notes recorded yet)"

        return check_explorer_notes

    def _seed_block(self, seed_type: Optional[str] = None) -> str:
        """The task's seed artifacts as markdown. With no seed_type, EVERY
        group in the library is included, one section per type."""
        types = [seed_type] if seed_type else (self.task.seed_types() or [None])
        parts = []
        for st in types:
            group = self.task.seeds(st)
            if not group:
                continue
            entries = "\n\n".join(f"### {n}\n```python\n{c.strip()}\n```"
                                  for n, c in group.items())
            parts.append(f"## Seed type: {st}\n{entries}" if st else entries)
        return "\n\n".join(parts) or "(no seeds available)"

    def _make_seed_library_tool(self):
        """Tool exposing the task's seed designs (replaces pasting them inline)."""
        pipeline = self

        @tool
        def view_seed_library(seed_type: str = "") -> str:
            """Return the library of known-good seed designs for this task.

            These are reference artifacts you may start from, adapt, or
            combine. By default the ENTIRE library is returned, grouped by
            seed type; pass `seed_type` to see just one group. Consult it
            before proposing your first candidate and whenever you want a
            proven pattern to build on.
            """
            return pipeline._seed_block(seed_type or None)

        return view_seed_library

    def _explorer_toolset(self) -> list:
        """Everything the explorer can call: injected research tools (web /
        docs / papers -- optional) plus the seed library."""
        return self._dedupe_tools(self.explorer_tools + [self.seed_library_tool])

    def _design_tools(self) -> list:
        """Everything the generate LLM can call: research tools, the seed
        library, the notes tool (iff the explorer is in the graph), and ALWAYS
        the evaluate tool."""
        tools = list(self.generate_tools) + [self.seed_library_tool]
        if self.use_explorer:
            tools.append(self.check_notes_tool)
        tools.append(self.evaluate_tool)
        return self._dedupe_tools(tools)

    @staticmethod
    def _dedupe_tools(tools) -> list:
        seen: dict = {}
        for t in tools:
            seen[t.name] = t
        return list(seen.values())

    # ------------------------------------------------------------------ helpers
    def _sys(self, agent_type: str, default: str) -> str:
        """Task system prompt for `agent_type`, falling back to `default`."""
        try:
            return self.task.system_prompt(agent_type)
        except KeyError:
            return default

    def _note(self, section: str, text: str) -> None:
        """Append a finding to the persistent notes file (research storage)."""
        with open(self.notes_path, "a", encoding="utf-8") as f:
            f.write(f"\n## {section}\n{text}\n")

    def _invoke_tool(self, tool_map: dict, call: dict,
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
        self.task.log(f"TOOL {call['name']} [tool {dt:.2f}s]", out[:1500])
        if note:
            self._note(f"TOOL {call['name']}", out[:2000])
        try:
            parsed = json.loads(out)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        return (ToolMessage(content=out, tool_call_id=call["id"], name=call["name"]),
                parsed)

    def _find_all_evals(self, messages):
        """Every (code, metrics) from the latest batch of `evaluate` tool calls.

        Skipped variants (cap exceeded / earlier failure) are excluded -- they
        were never actually run, so they must not count as iterations.
        """
        trigger = None
        for msg in reversed(messages):
            if _has_tool_calls(msg):
                trigger = msg
                break
        if trigger is None:
            return []
        results_by_id = {m.tool_call_id: m.content
                         for m in messages if isinstance(m, ToolMessage)}
        evals = []
        for call in trigger.tool_calls:
            if call["name"] != self.evaluate_tool_name:
                continue
            code = call["args"].get("code")
            raw = results_by_id.get(call["id"])
            try:
                metrics = json.loads(raw) if raw is not None else {"ok": False, "error": "no result"}
            except (json.JSONDecodeError, TypeError):
                metrics = {"ok": False, "error": str(raw)}
            if isinstance(metrics, dict) and "skipped" in metrics:
                continue
            evals.append((code, metrics))
        return evals

    def _run_with_tools(self, llm, tools, messages):
        """Invoke an LLM, servicing any tool calls, until it returns plain text.

        Used by the review node (its own small tool loop, separate from the
        graph's tool-executor nodes)."""
        bound = llm.bind_tools(tools) if tools else llm
        tool_map = {t.name: t for t in tools}
        convo = list(messages)
        resp = None
        dt = 0.0
        for _ in range(self.max_tool_rounds + 1):
            t0 = time.perf_counter()
            resp = bound.invoke(convo)
            dt = time.perf_counter() - t0
            convo.append(resp)
            if not _has_tool_calls(resp):
                return resp, convo, dt
            for call in resp.tool_calls:
                msg, _ = self._invoke_tool(tool_map, call)
                convo.append(msg)
        return resp, convo, dt

    # -------------------------------------------------------------------- nodes
    def explorer_node(self, state: PipelineState) -> dict:
        default = ("You are a research assistant for an algorithm-discovery task. "
                   "Use the available tools to gather relevant background "
                   "(documentation, prior work, expert advice, the seed design "
                   "library). Take concise, actionable notes for the designer. "
                   "When you have enough, stop calling tools and respond with a "
                   "short notes summary.")
        bound = self.explorer_llm.bind_tools(self._explorer_toolset())
        t0 = time.perf_counter()
        resp = bound.invoke([SystemMessage(content=self._sys("explore", default))]
                            + state["explore_messages"])
        dt = time.perf_counter() - t0
        rounds = state.get("explore_rounds", 0) + 1

        # accumulate EVERY round's written notes (not just the final summary);
        # this full text is what the designer will see in its system prompt.
        notes = state.get("notes", "")
        text = _text_of(resp).strip()
        if text:
            notes = f"{notes}\n\n{text}".strip() if notes else text
            self._note(f"EXPLORE (round {rounds})", text)

        out: dict = {"explore_messages": [resp], "explore_rounds": rounds,
                     "notes": notes}
        # hand off exactly when we will NOT keep researching
        handing_off = (not _has_tool_calls(resp)) or (rounds >= self.max_explore_rounds)
        if handing_off:
            out["notes"] = notes or "(no notes produced)"
            self.task.log(f"EXPLORE handoff [llm {dt:.2f}s]", out["notes"])
        else:
            self.task.log(f"EXPLORE (round {rounds}) [llm {dt:.2f}s]", text)
        return out

    def explore_tools_node(self, state: PipelineState) -> dict:
        """Tool executor for the explorer's channel (research calls only)."""
        last = state["explore_messages"][-1]
        tool_map = {t.name: t for t in self._explorer_toolset()}
        out = [self._invoke_tool(tool_map, call, note=True)[0]
               for call in last.tool_calls]
        return {"explore_messages": out}

    def generate_node(self, state: PipelineState) -> dict:
        # the explorer's FULL written notes ride in the system prompt: present
        # from the very first generate call, resent on every call, but never
        # duplicated into the accumulating message thread. Raw tool dumps stay
        # out (they can be huge and low-signal) -- those are one
        # `check_explorer_notes` call away.
        sys_content = self.task.system_prompt("generate")
        notes = (state.get("notes") or "").strip()
        if notes:
            sys_content += (
                "\n\n# Research notes from the explorer\n" + notes +
                "\n\n(Raw tool findings behind these notes: call "
                "`check_explorer_notes`.)")
        bound = self.generate_llm.bind_tools(self._design_tools())
        t0 = time.perf_counter()
        resp = bound.invoke([SystemMessage(content=sys_content)]
                            + state["messages"])
        dt = time.perf_counter() - t0
        self.task.log(f"GENERATE (round {state.get('iteration', 0)}) [llm {dt:.2f}s]",
                      _text_of(resp))
        out: dict = {"messages": [resp]}
        code = _extract_code(resp.content)
        if code:
            out["current_code"] = code
        if _has_tool_calls(resp):
            out["gen_no_tool"] = 0
        else:
            out["gen_no_tool"] = state.get("gen_no_tool", 0) + 1
            # A plain reply AFTER evaluating means "I'm satisfied" -> the router
            # sends us to deploy, which ships best-so-far. Only nudge a model
            # that talks without ever having evaluated anything.
            if not state.get("last_metrics"):
                out["messages"] = [resp, HumanMessage(content=(
                    "You have not evaluated any candidate yet. Reply by calling "
                    f"`{self.evaluate_tool_name}` with your candidate code "
                    f"(up to {self.max_variants_per_turn} variants per turn, "
                    "most promising first)."))]
        return out

    def gen_tools_node(self, state: PipelineState) -> dict:
        """Tool executor for the design thread.

        Research calls all run. Evaluate calls run SEQUENTIALLY in a for loop
        that (a) caps at max_variants_per_turn and (b) terminates early when a
        variant comes back invalid (error / not ok). Crucially, every skipped
        tool_call id STILL receives a ToolMessage -- providers reject a chat
        where a tool call has no result, so this is what keeps a multi-call
        turn from corrupting the state.
        """
        last = state["messages"][-1]
        tool_map = {t.name: t for t in self._design_tools()}
        out_messages = []
        ran, failed = 0, False
        for call in last.tool_calls:
            if call["name"] != self.evaluate_tool_name:
                out_messages.append(self._invoke_tool(tool_map, call)[0])
                continue
            if failed:
                skip_reason = "an earlier variant in this batch failed; fix it before testing more"
            elif ran >= self.max_variants_per_turn:
                skip_reason = f"max {self.max_variants_per_turn} variants per turn"
            else:
                msg, parsed = self._invoke_tool(tool_map, call)
                out_messages.append(msg)
                ran += 1
                if not (isinstance(parsed, dict) and parsed.get("ok")):
                    failed = True
                continue
            self.task.log(f"TOOL {call['name']} (skipped)", skip_reason)
            out_messages.append(ToolMessage(
                content=json.dumps({"ok": False, "skipped": skip_reason}),
                tool_call_id=call["id"], name=call["name"]))
        return {"messages": out_messages}

    def evaluate_node(self, state: PipelineState) -> dict:
        """DETERMINISTIC record step (no LLM, no re-run).

        The generate LLM already chose what code to execute -- its evaluate
        tool calls carried the code, and gen_tools ran them. This node folds
        every result of that batch into best-so-far and summarizes the batch
        for the reviewer and the designer.
        """
        evals = self._find_all_evals(state["messages"])
        if not evals:
            evals = [(state.get("current_code", ""),
                      {"ok": False, "error": "no evaluation result found"})]
        best = dict(state.get("best") or {})
        it = state.get("iteration", 0)
        batch_best = (-1.0, None, evals[-1][1])   # (score, code, metrics)
        summary = []
        for code, metrics in evals:
            it += 1
            score = metrics.get(self.target_metric, -1.0) if metrics.get("ok") else -1.0
            if score > best.get(self.target_metric, -1.0):
                best = {"code": code, self.target_metric: score, "metrics": metrics}
            if score > batch_best[0]:
                batch_best = (score, code, metrics)
            summary.append(json.dumps(metrics))
        self.task.log(f"EVALUATE (round {it}, +{len(evals)} variant(s))",
                      "\n".join(summary))
        out = {"last_metrics": batch_best[2], "best": best, "iteration": it,
               "messages": [HumanMessage(content=(
                   f"Evaluation results for this batch ({len(evals)} variant(s)):\n"
                   + "\n".join(summary)))]}
        if batch_best[1]:
            out["current_code"] = batch_best[1]
        return out

    def review_node(self, state: PipelineState) -> dict:
        """Reviewer with its OWN message thread: it remembers every candidate
        it has critiqued and what it already suggested, so feedback does not
        repeat. Only the distilled feedback string enters the design thread."""
        m = state.get("last_metrics", {})
        code = state.get("current_code", "")
        prompt = (f"Round {state.get('iteration', 0)}. "
                  f"Best candidate metrics this round: {json.dumps(m)}\n"
                  f"Code:\n```python\n{code}\n```\n"
                  f"Give one concrete, actionable improvement (or the fix if it "
                  f"errored). Avoid repeating advice you already gave.")
        # the seed library rides in the SYSTEM message: the reviewer sees it on
        # every call, but it is never appended to the accumulating review
        # thread, so it is not duplicated round after round.
        review_system = (self.task.system_prompt("review")
                         + "\n\n# Seed library (known-good reference designs)\n"
                         + self._seed_block())
        resp, _, dt = self._run_with_tools(
            self.review_llm, self.review_tools,
            [SystemMessage(content=review_system)]
            + state.get("review_messages", [])
            + [HumanMessage(content=prompt)],
        )
        note = _text_of(resp)
        self.task.log(f"REVIEW (round {state.get('iteration', 0)}) [llm {dt:.2f}s]", note)
        return {"review": note,
                # reviewer's own memory: the question + its distilled answer
                # (a fresh AIMessage, so no dangling tool-call pairs persist)
                "review_messages": [HumanMessage(content=prompt),
                                    AIMessage(content=note)],
                "messages": [HumanMessage(content=f"Reviewer feedback: {note}")]}

    def deploy_node(self, state: PipelineState) -> dict:
        best = state.get("best") or {}
        extra = (f"\nAdditional user instructions for deployment:\n{self.deploy_prompt}"
                 if self.deploy_prompt else "")
        t0 = time.perf_counter()
        summary_resp = self.deploy_llm.invoke([
            SystemMessage(content=self.task.system_prompt("deploy")),
            HumanMessage(content=(
                f"The search finished after {state.get('iteration', 0)} variants. "
                f"Best artifact and stats:\n{json.dumps(best, indent=2)}\n"
                f"Summarize what was designed, the final stats, and the rationale."
                f"{extra}")),
        ])
        summary = _text_of(summary_resp)
        self.task.log(f"DEPLOY summary [llm {time.perf_counter() - t0:.2f}s]", summary)

        translated = None
        if self.translate_llm and self.translate_spec and best.get("code"):
            t0 = time.perf_counter()
            tr = self.translate_llm.invoke([
                HumanMessage(content=(f"{self.translate_spec}\n\nArtifact:\n"
                                      f"```python\n{best['code']}\n```")),
            ])
            translated = _text_of(tr)
            self.task.log(f"DEPLOY translation [llm {time.perf_counter() - t0:.2f}s]",
                          translated)

        return {"messages": [summary_resp],
                "review": summary,
                "last_metrics": {**state.get("last_metrics", {}),
                                 "summary": summary, "translated": translated}}

    # --------------------------------------------------------------- edges
    def _done(self, state: PipelineState) -> bool:
        """Deterministic stop check: budget exhausted or target reached."""
        if state.get("iteration", 0) >= self.max_iters:
            return True
        best = state.get("best") or {}
        return best.get(self.target_metric, -1.0) >= self.target_value - 1e-9

    def _route_after_explorer(self, state: PipelineState) -> str:
        last = state["explore_messages"][-1]
        if _has_tool_calls(last) and state.get("explore_rounds", 0) < self.max_explore_rounds:
            return "explore_tools"
        return "generate"          # done researching -> design

    def _route_after_generate(self, state: PipelineState) -> str:
        if _has_tool_calls(state["messages"][-1]):
            return "gen_tools"      # research or evaluate call(s) -> executor
        if state.get("last_metrics"):
            return "deploy"         # plain reply after evaluating -> ship best-so-far
        if state.get("gen_no_tool", 0) >= self.max_gen_no_tool:
            return "deploy"         # never evaluated and won't -> wrap up
        return "generate"           # nudged; let it try again

    def _route_after_gen_tools(self, state: PipelineState) -> str:
        for msg in reversed(state["messages"]):
            if _has_tool_calls(msg):
                if any(c["name"] == self.evaluate_tool_name for c in msg.tool_calls):
                    return "evaluate"   # evaluated variants -> record them
                break
        return "generate"          # pure research batch -> back to the designer

    def _route_after_evaluate(self, state: PipelineState) -> str:
        return "deploy" if self._done(state) else "review"

    def _route_after_review(self, state: PipelineState) -> str:
        return "deploy" if self._done(state) else "generate"

    # --------------------------------------------------------------- graph
    def build_graph(self):
        g = StateGraph(PipelineState)
        g.add_node("generate", self.generate_node)
        g.add_node("gen_tools", self.gen_tools_node)
        g.add_node("evaluate", self.evaluate_node)
        g.add_node("review", self.review_node)
        g.add_node("deploy", self.deploy_node)

        if self.use_explorer:
            # the explorer always has at least the seed-library tool, so its
            # ToolNode is always wired in
            g.add_node("explorer", self.explorer_node)
            g.add_node("explore_tools", self.explore_tools_node)
            g.add_edge(START, "explorer")
            g.add_conditional_edges("explorer", self._route_after_explorer,
                                    {"explore_tools": "explore_tools",
                                     "generate": "generate"})
            g.add_edge("explore_tools", "explorer")
        else:
            g.add_edge(START, "generate")            # explorer toggled off

        g.add_conditional_edges("generate", self._route_after_generate,
                                {"gen_tools": "gen_tools",
                                 "generate": "generate", "deploy": "deploy"})
        g.add_conditional_edges("gen_tools", self._route_after_gen_tools,
                                {"evaluate": "evaluate", "generate": "generate"})
        g.add_conditional_edges("evaluate", self._route_after_evaluate,
                                {"review": "review", "deploy": "deploy"})
        g.add_conditional_edges("review", self._route_after_review,
                                {"generate": "generate", "deploy": "deploy"})
        g.add_edge("deploy", END)
        return g.compile()

    # --------------------------------------------------------------- run
    def run(self, extra_instructions: str = "") -> dict:
        design_first = HumanMessage(content=(
            "Design and iteratively improve the artifact. Start by calling "
            "`view_seed_library` to see known-good reference designs you may "
            f"adapt or combine. {extra_instructions}\n"
            f"Call the `{self.evaluate_tool_name}` tool on each candidate to "
            f"score it -- up to {self.max_variants_per_turn} variants per turn, "
            "most promising first (if one errors, the rest of the turn is "
            "skipped). When you are satisfied, reply WITHOUT calling any tool "
            "and the best candidate so far will be deployed."))

        init_state: dict = {"messages": [design_first], "review_messages": [],
                            "notes": "", "iteration": 0, "best": {},
                            "gen_no_tool": 0}
        if self.use_explorer:
            init_state["explore_messages"] = [HumanMessage(content=(
                "Research this design problem so the designer can start well "
                f"informed. {extra_instructions}"))]
            init_state["explore_rounds"] = 0

        self.task.log("RUN CONFIG",
                      f"{self.task.describe()}\n"
                      f"explorer={'ON' if self.use_explorer else 'OFF'} "
                      f"explorer_tools={[t.name for t in self._explorer_toolset()]}\n"
                      f"design_tools={[t.name for t in self._design_tools()]} "
                      f"review_tools={[t.name for t in self.review_tools]}\n"
                      f"max_iters={self.max_iters} "
                      f"target={self.target_metric}>={self.target_value} "
                      f"max_variants_per_turn={self.max_variants_per_turn} "
                      f"entry_point={self.entry_point}")
        run_t0 = time.perf_counter()
        app = self.build_graph()
        final = app.invoke(init_state, {"recursion_limit": 100})
        self.task.log("TOTAL RUNTIME",
                      f"{time.perf_counter() - run_t0:.2f}s over "
                      f"{final.get('iteration', 0)} variants")
        return {"best": final.get("best", {}),
                "summary": final.get("review", ""),
                "notes": final.get("notes", ""),
                "translated": final.get("last_metrics", {}).get("translated"),
                "iterations": final.get("iteration", 0),
                "log_path": self.task.log_path,
                "notes_path": self.notes_path}
