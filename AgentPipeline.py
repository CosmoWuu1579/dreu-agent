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

TRIALS vs VARIANTS (two separate evaluations):

  trial   : ONE `evaluate` tool call made by the designer -- scratch testing.
            Runs the evaluator, result comes straight back to generate so it
            can iterate. Trials never touch best-so-far. Budget of
            max_session_trials (default 3) per design session.
  variant : the OFFICIAL score. When the designer submits (a plain reply
            after >=1 trial, or the budget forces it), the evaluate node
            RE-RUNS the evaluator on the submitted code and records the
            result as variant #K. Only variants update best-so-far and
            `iteration` (max_iters counts variants, not trials).

  What gets submitted: when the budget is spent, the LAST trial's result still
  goes back to generate with an instruction to analyze everything and present
  its final design; the code block in that reply is the submission (no block
  -> the session's BEST trial; ties go to the LATER trial). Only a designer
  that keeps calling evaluate after being asked to submit is force-submitted.
  As a final safety net, if the best trial of the whole run still beats the
  best official variant at deploy time, deploy officially re-evaluates it
  first -- the best circuit ever tested is never lost.

   START -> [explorer <--> explore_tools]        (optional: use_explorer=False
              |  (hand off notes)                 removes it from the graph)
              v
           generate <--> gen_tools --(trial results, incl. the last one
              ^  |              |     + "budget spent, choose")--> generate
              |  | (plain reply | (kept calling evaluate after being
              |  |  after >=1   |  asked to submit -> forced)
              |  |  trial)      v
              |  +---------> evaluate    OFFICIAL: re-run the evaluator on
              |                 |        the submitted design -> variant #K
              |          (not done) | (done)
              +---- review <--------+   |
              ^        |                v
              +(feedback)            deploy -> END

  explorer : optional ENTRY. LLM with research tools (web search / docs /
             papers -- all injected, all optional) plus `view_seed_library`.
             Its written notes from EVERY round accumulate into `notes`, which
             is injected into the generate LLM's SYSTEM prompt (present from
             the first design call onward). Raw tool outputs go only to the
             notes file -- the explorer's private record -- which the designer
             can read via the `check_explorer_notes` tool.
  generate : LLM with research tools + `view_seed_library` +
             `check_explorer_notes` (iff the explorer exists) + the trial
             evaluate tool. It tests circuits BY CALLING evaluate -- the
             tool-call args carry the code -- iterating trial by trial. A
             plain reply after >=1 trial means "this is my submission".
  gen_tools: custom tool executor for the design thread. Trial evaluate calls
             run in a for loop that STOPS at the first invalid (errored)
             trial; every tool_call id still receives a ToolMessage (skipped
             ones get an explicit marker) so the chat state stays valid.
  evaluate : OFFICIAL evaluation -- no LLM. Re-runs the evaluator on the
             submitted design and records it as the next variant. The ONLY
             place best-so-far / iteration are updated. Resets the session.
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
    # --- message channels, one per agent (never mixed) -----------------------
    messages: Annotated[list, add_messages]
    #   the DESIGN thread: generate's conversation -- its tool calls, the tool
    #   results, official-evaluation notices, and reviewer feedback.
    explore_messages: Annotated[list, add_messages]
    #   the explorer's private research thread (tool calls + results). Never
    #   seen by the designer; only `notes` crosses over.
    review_messages: Annotated[list, add_messages]
    #   the reviewer's private thread: one (prompt, answer) pair per round, so
    #   it remembers what it already advised. Tool plumbing is not persisted.

    # --- explorer output ------------------------------------------------------
    notes: str
    #   the explorer's accumulated written notes (every round). Injected into
    #   generate's SYSTEM prompt on every call. Raw tool dumps stay in the
    #   notes file (readable via the check_explorer_notes tool).

    # --- design-session bookkeeping (reset by the evaluate node) --------------
    session_trials: int
    #   how many trial evaluations the designer has used in the CURRENT
    #   session. Routing reads this: >0 + plain reply -> submit to the
    #   evaluate node. At >= max_session_trials the designer is ASKED to
    #   present its final design (one analysis turn); only if it then calls
    #   evaluate again is submission forced (see force_official). Reset to 0
    #   after each official evaluation.
    current_code: str
    #   the design currently on the table: the BEST-scoring trial of this
    #   session, overridden by the code block in generate's reply if it
    #   contains one. THIS is what the evaluate node officially scores.
    session_best: dict
    #   {"score", "code", "trial"} of this session's best trial (drives
    #   current_code); on equal scores the LATER trial wins. Also the FALLBACK
    #   submission if the designer's chosen code errors in official
    #   evaluation. Reset by the evaluate node.
    force_official: bool
    #   set when the designer was asked to submit (budget spent) but called
    #   evaluate again anyway -- the router then forces official evaluation.
    #   Reset by the evaluate node.
    best_trial: dict
    #   {"score", "code", "trial", "metrics"} of the best trial of the WHOLE
    #   run, official or not. If it still beats the official best at deploy
    #   time, deploy re-evaluates it officially (salvage) before summarizing.
    session_research: int
    #   non-evaluate ("research") tool calls the designer has used this
    #   session (docs / papers / seed library / notes). At
    #   max_session_research further research calls are skipped with an
    #   out-of-budget message; each executed call's result is annotated with
    #   the remaining count. Reset by the evaluate node.
    gen_no_tool: int
    #   consecutive unproductive generate turns: a reply that needed a
    #   corrective nudge (no trial run yet, new design presented without
    #   calling evaluate, malformed tool call), or a tool batch in which
    #   EVERY call was skipped (budgets exhausted). At max_gen_no_tool we
    #   stop indulging it: submit the session's best if any trial ran,
    #   otherwise deploy.

    # --- official results (ONLY the evaluate node writes these) ---------------
    iteration: int
    #   official variants recorded so far. max_iters caps this, not trials.
    last_metrics: dict
    #   metrics of the most recent official variant (what review critiques).
    best: dict
    #   best official variant so far: {code, <target_metric>, metrics, variant}.
    review: str
    #   the latest reviewer feedback (after deploy: the final summary).
    explore_rounds: int
    #   count of explorer research rounds so far. Up to max_explore_rounds
    #   rounds use research tools; then one extra turn writes final notes and
    #   hands off (so the cap counts research rounds, not the notes turn).


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


class AgentPipeline:
    def __init__(
        self,
        task: DiscoveryTask,
        generate_llm,
        *,
        use_explorer: bool = True,               # toggle the explorer subgraph
        use_notes_tool: bool = False,            # give generate a re-read tool
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
        max_tool_rounds: int = 3,                # review: max tool CALLS per round
        max_session_trials: int = 3,             # trial evaluations per design session
        max_session_research: int = 4,           # non-evaluate tool calls per session
    ):
        self.task = task
        self.evaluate_kwargs = evaluate_kwargs or {}
        self.generate_llm = generate_llm
        self.explorer_llm = explorer_llm or generate_llm
        self.review_llm = review_llm or generate_llm
        self.deploy_llm = deploy_llm or generate_llm
        self.translate_llm = translate_llm
        self.use_explorer = use_explorer
        # the notes tool only makes sense when the explorer runs; it is also
        # redundant with the notes already in generate's system prompt, so it
        # is opt-in (default off to save tool calls / context)
        self.use_notes_tool = use_notes_tool and use_explorer
        self.explorer_tools = explorer_tools or []
        self.generate_tools = generate_tools or []
        self.review_tools = review_tools or []
        # global variant counter: every evaluate call gets a number so logs,
        # review feedback, plot filenames, and the final report line up
        self._eval_count = 0
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
        self.max_session_trials = max_session_trials
        self.max_session_research = max_session_research
        # a persistent side-file where explorer notes + tool findings accumulate
        # (lives inside the task's per-run folder, next to the transcript)
        self.notes_path = os.path.join(task.run_dir, "notes.md")
        # the explorer's final handoff notes; what check_explorer_notes returns
        # (that tool is wired into generate only when use_notes_tool is on)
        self._explorer_notes = ""
        self.check_notes_tool = self._make_check_notes_tool()
        # seed reference designs, exposed as a tool (explorer + generate)
        self.seed_library_tool = self._make_seed_library_tool()
        # last system prompt logged per role, so the transcript records each
        # system prompt once on first use and again whenever it changes
        self._logged_systems: dict = {}

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
        """Default evaluate tool: runs task.evaluate and returns metrics as JSON.

        Every call is stamped with a global `trial` number (also forwarded as
        `iteration=` to the evaluator, which e.g. tags plot filenames), so
        every artifact of the run traces back to the trial that produced it."""
        task = self.task
        pipeline = self

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

    def _make_check_notes_tool(self):
        """Tool letting the designer re-read the explorer's distilled notes."""
        pipeline = self

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
        library, the notes tool (only if use_notes_tool), and ALWAYS the
        evaluate tool."""
        tools = list(self.generate_tools) + [self.seed_library_tool]
        if self.use_notes_tool:
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

    def _log_system(self, role: str, content: str) -> None:
        """Log a role's system prompt on first use and whenever it changes
        (e.g. generate's gains the explorer notes after the handoff)."""
        if self._logged_systems.get(role) != content:
            self._logged_systems[role] = content
            self.task.log(f"SYSTEM PROMPT ({role})", content)

    def _note(self, section: str, text: str) -> None:
        """Append a finding to the persistent notes file (research storage)."""
        os.makedirs(os.path.dirname(self.notes_path), exist_ok=True)
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
        # log the INPUTS too (e.g. the exact code a trial evaluated), in full --
        # the transcript is the record of what was actually tested
        args_lines = []
        for k, v in (call.get("args") or {}).items():
            v_str = v if isinstance(v, str) else json.dumps(v)
            args_lines.append(f"{k}:\n{v_str}" if "\n" in v_str else f"{k}: {v_str}")
        args_view = "\n".join(args_lines) or "(no args)"
        self.task.log(f"TOOL {call['name']} [tool {dt:.2f}s]",
                      f"--- args ---\n{args_view}\n--- result ---\n{out[:]}")
        if note:
            self._note(f"TOOL {call['name']}", out[:])
        try:
            parsed = json.loads(out)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        return (ToolMessage(content=out, tool_call_id=call["id"], name=call["name"]),
                parsed)

    def _run_with_tools(self, llm, tools, messages):
        """Invoke an LLM, servicing any tool calls, until it returns plain text.

        Used by the review node (its own small tool loop, separate from the
        graph's tool-executor nodes).

        Every tool result is appended to the conversation BEFORE the next
        invoke, so the final plain-text reply is produced with all of them in
        context -- nothing gathered along the way is hidden from it. At most
        max_tool_rounds tool CALLS are executed in total per review round
        (calls_used is a local: this whole loop runs once per official
        variant, so the budget resets naturally each round); calls beyond the
        budget are answered with an out-of-budget message instead of running.
        If the model is still calling tools when the rounds run out, it is
        explicitly asked to stop and summarize, so the final answer never
        comes back as an empty tool-call stub."""
        bound = llm.bind_tools(tools) if tools else llm
        tool_map = {t.name: t for t in tools}
        convo = list(messages)
        dt = 0.0
        calls_used = 0
        for _ in range(self.max_tool_rounds):
            t0 = time.perf_counter()
            resp = bound.invoke(convo)
            dt += time.perf_counter() - t0
            convo.append(resp)
            if not _has_tool_calls(resp):
                return resp, convo, dt
            # log what the model asked its counsel, not just the answers
            asks = "; ".join(f"{c['name']}({json.dumps(c['args'])})"
                             for c in resp.tool_calls)
            self.task.log("TOOL REQUEST (review)",
                          (_text_of(resp).strip() + "\n-> " + asks).strip())
            for call in resp.tool_calls:
                if calls_used >= self.max_tool_rounds:
                    skip = (f"tool budget ({self.max_tool_rounds} calls) "
                            "exhausted; give your final answer now")
                    self.task.log(f"TOOL {call['name']} (skipped)", skip)
                    convo.append(ToolMessage(
                        content=json.dumps({"ok": False, "skipped": skip}),
                        tool_call_id=call["id"], name=call["name"]))
                    continue
                msg, _ = self._invoke_tool(tool_map, call)
                calls_used += 1
                convo.append(msg)
        # budget exhausted while still calling tools: demand a final summary
        # so everything learned from the tool calls makes it into the answer
        demand = ("Tool budget exhausted. Do NOT call any more tools. Summarize "
                  "your final answer now, incorporating everything you learned "
                  "above.")
        self.task.log("WARNING (review)", demand)
        convo.append(HumanMessage(content=demand))
        t0 = time.perf_counter()
        resp = bound.invoke(convo)
        dt += time.perf_counter() - t0
        convo.append(resp)
        return resp, convo, dt

    # -------------------------------------------------------------------- nodes
    def explorer_node(self, state: PipelineState) -> dict:
        default = ("You are a research assistant for an algorithm-discovery task. "
                   "Use the available tools to gather relevant background "
                   "(documentation, prior work, expert advice, the seed design "
                   "library). Take concise, actionable notes for the designer. "
                   "When you have enough, stop calling tools and respond with a "
                   "short notes summary.")
        sys_content = self._sys("explore", default)
        self._log_system("explorer", sys_content)
        bound = self.explorer_llm.bind_tools(self._explorer_toolset())
        t0 = time.perf_counter()
        resp = bound.invoke([SystemMessage(content=sys_content)]
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
        # hand off when the explorer stops researching on its own, OR after it
        # has used all max_explore_rounds research rounds AND been given one
        # extra turn to write notes (that notes turn is round max+1).
        handing_off = (not _has_tool_calls(resp)) or (rounds > self.max_explore_rounds)
        if handing_off:
            out["notes"] = notes or "(no notes produced)"
            # what the designer re-reads via check_explorer_notes (clean notes,
            # not the raw per-tool findings that also land in notes.md)
            self._explorer_notes = out["notes"]
            self.task.log(f"EXPLORE handoff [llm {dt:.2f}s]", out["notes"])
        else:
            self.task.log(f"EXPLORE (round {rounds}) [llm {dt:.2f}s]", text)
        return out

    def explore_tools_node(self, state: PipelineState) -> dict:
        """Tool executor for the explorer's channel (research calls only).

        Runs BETWEEN explorer turns. Once all max_explore_rounds research
        rounds are used, it appends a wrap-up warning so the explorer spends
        its one extra (notes) turn writing final notes rather than making tool
        calls that would be dropped when the handoff is forced."""
        last = state["explore_messages"][-1]
        tool_map = {t.name: t for t in self._explorer_toolset()}
        out = [self._invoke_tool(tool_map, call, note=True)[0]
               for call in last.tool_calls]
        if state.get("explore_rounds", 0) >= self.max_explore_rounds:
            warning = ("You have used all your research rounds. Do NOT call "
                       "any more tools -- reply with your complete, actionable "
                       "notes summary for the designer.")
            self.task.log("WARNING (explorer)", warning)
            out.append(HumanMessage(content=warning))
        return {"explore_messages": out}

    def generate_node(self, state: PipelineState) -> dict:
        # the explorer's FULL written notes ride in the system prompt: present
        # from the very first generate call, resent on every call, but never
        # duplicated into the accumulating message thread. Raw tool dumps stay
        # out (they can be huge and low-signal) -- those are one
        # `check_explorer_notes` call away.
        #
        # The system prompt stays STATIC (task instructions + explorer notes).
        # The dynamic "best design so far / build on it" reminder is NOT added
        # here -- it lives in the message thread (injected by review_node at
        # session start), so it appears in correct chronological order (right
        # after the variant it names, not at position 0 before that variant's
        # own messages) and does not invalidate the cached system prompt.
        sys_content = self.task.system_prompt("generate")
        notes = (state.get("notes") or "").strip()
        if notes:
            sys_content += "\n\n# Research notes from the explorer\n" + notes
            if self.use_notes_tool:
                sys_content += ("\n\n(You can re-read these notes anytime with "
                                "`check_explorer_notes`.)")
        self._log_system("generate", sys_content)
        bound = self.generate_llm.bind_tools(self._design_tools())
        t0 = time.perf_counter()
        resp = bound.invoke([SystemMessage(content=sys_content)]
                            + state["messages"])
        dt = time.perf_counter() - t0
        self.task.log(f"GENERATE (round {state.get('iteration', 0)}) [llm {dt:.2f}s]",
                      _text_of(resp))
        out: dict = {"messages": [resp]}
        if _has_tool_calls(resp):
            # gen_tools decides whether this batch is productive (it may skip
            # every call on exhausted budgets), so it owns the strike counter
            return out

        # Plain reply: either a real submission, or a slip that needs a
        # corrective nudge. A model sometimes MEANS to call the tool but emits
        # a malformed call (unclosed code fence / tool-XML as text) -- that
        # must not silently end the session as a submission.
        text = _text_of(resp)
        block = _extract_code(text)
        trials_used = state.get("session_trials", 0)
        budget_left = trials_used < self.max_session_trials
        sess_best_code = (state.get("session_best") or {}).get("code")

        nudge = None
        if trials_used == 0:
            nudge = ("You have not run any trial this session. Reply by "
                     f"calling `{self.evaluate_tool_name}` with your "
                     f"candidate code (budget: {self.max_session_trials} "
                     "trials per session, most promising first).")
        elif budget_left and block and block not in (
                sess_best_code, state.get("current_code")):
            # a NEW, untested design presented as text: trial it, don't
            # silently submit something else
            nudge = ("You presented a new design but did not call the "
                     f"`{self.evaluate_tool_name}` tool, so it was NOT "
                     f"tested. You have "
                     f"{self.max_session_trials - trials_used} trial(s) "
                     f"left: call `{self.evaluate_tool_name}` with this "
                     "exact code to test it, or reply with no code block to "
                     "submit your best trial so far.")
        elif budget_left and not block and self.entry_point in text:
            # code-like text but no complete fenced block: usually a
            # truncated or malformed tool call
            nudge = ("Your reply contains code but no complete ```python "
                     "block could be parsed (this often means a malformed "
                     f"tool call). Call `{self.evaluate_tool_name}` with "
                     "the complete code to test it.")

        if nudge:
            out["gen_no_tool"] = state.get("gen_no_tool", 0) + 1
            self.task.log("WARNING (generate)", nudge)
            out["messages"] = [resp, HumanMessage(content=nudge)]
        else:
            out["gen_no_tool"] = 0
            if block:
                # an explicit code block in a SUBMISSION overrides the
                # session's best trial as the officially evaluated design
                out["current_code"] = block
        return out

    def gen_tools_node(self, state: PipelineState) -> dict:
        """Tool executor for the design thread.

        Research calls all run. Trial evaluate calls run SEQUENTIALLY in a for
        loop that (a) caps at the session budget (max_session_trials) and
        (b) terminates early when a trial comes back invalid (error / not ok).
        Crucially, every skipped tool_call id STILL receives a ToolMessage --
        providers reject a chat where a tool call has no result, so this is
        what keeps a multi-call turn from corrupting the state.

        Bumps state["session_trials"] per executed trial and keeps
        state["session_best"] / state["best_trial"] / state["current_code"]
        pointing at the best-scoring trial (session / whole run) -- so the
        default submission is the session's BEST design, not its latest.
        """
        last = state["messages"][-1]
        tool_map = {t.name: t for t in self._design_tools()}
        trials = state.get("session_trials", 0)
        n_before = trials
        research = state.get("session_research", 0)
        research_before = research
        sess_best = dict(state.get("session_best") or {})
        run_best = dict(state.get("best_trial") or {})
        out_messages = []
        failed = False
        executed_any = False
        for call in last.tool_calls:
            if call["name"] != self.evaluate_tool_name:
                # research call (docs / papers / seeds / notes): budgeted per
                # session, and every result tells the model what's left
                if research >= self.max_session_research:
                    skip_reason = (
                        f"research budget ({self.max_session_research} "
                        "non-evaluate calls per session) exhausted; run a "
                        f"trial with `{self.evaluate_tool_name}` or reply "
                        "without tools to submit")
                    self.task.log(f"TOOL {call['name']} (skipped)", skip_reason)
                    out_messages.append(ToolMessage(
                        content=json.dumps({"ok": False, "skipped": skip_reason}),
                        tool_call_id=call["id"], name=call["name"]))
                else:
                    msg, _ = self._invoke_tool(tool_map, call)
                    research += 1
                    executed_any = True
                    left = self.max_session_research - research
                    out_messages.append(ToolMessage(
                        content=(f"{msg.content}\n[research calls left this "
                                 f"session: {left}]"),
                        tool_call_id=call["id"], name=call["name"]))
                continue
            if failed:
                skip_reason = ("an earlier trial in this batch failed; "
                               "fix it before testing more")
            elif trials >= self.max_session_trials:
                skip_reason = (f"session budget of {self.max_session_trials} "
                               "trials reached; reply WITHOUT tools to submit "
                               "your design for official evaluation")
            else:
                msg, parsed = self._invoke_tool(tool_map, call)
                out_messages.append(msg)
                trials += 1
                executed_any = True
                metrics = parsed if isinstance(parsed, dict) else {}
                code = call["args"].get("code", "")
                if metrics.get("ok"):
                    score = metrics.get(self.target_metric, -1.0)
                    # >= so equal scores go to the LATER trial
                    if score >= sess_best.get("score", -1.0):
                        sess_best = {"score": score, "code": code,
                                     "trial": metrics.get("trial")}
                    if score >= run_best.get("score", -1.0):
                        run_best = {"score": score, "code": code,
                                    "trial": metrics.get("trial"),
                                    "metrics": metrics}
                else:
                    failed = True
                continue
            self.task.log(f"TOOL {call['name']} (skipped)", skip_reason)
            out_messages.append(ToolMessage(
                content=json.dumps({"ok": False, "skipped": skip_reason}),
                tool_call_id=call["id"], name=call["name"]))
        # budget JUST spent: hand the last result back to the designer with an
        # explicit ask, so it can analyze trial N before choosing what to submit
        if n_before < self.max_session_trials <= trials:
            ask = (f"Trial budget spent ({trials}/{self.max_session_trials}). "
                   "Analyze all your trial results, then reply WITHOUT tool "
                   "calls and present the design you want to submit as a "
                   "```python block. If you don't include one, your best trial "
                   f"(trial {sess_best.get('trial')}) is submitted.")
            self.task.log("WARNING (generate)", ask)
            out_messages.append(HumanMessage(content=ask))
        # make the routing visible in the transcript (mirrors trial logging)
        if research > research_before:
            dest = ("research budget spent -> run a trial or submit"
                    if research >= self.max_session_research
                    else "back to generate for more research or a trial")
            self.task.log(
                f"SESSION research {research}/{self.max_session_research}", dest)
        if trials > n_before:
            dest = ("budget spent -> designer asked for its final design"
                    if trials >= self.max_session_trials
                    else "back to generate for more trials")
            self.task.log(f"SESSION trials {trials}/{self.max_session_trials}",
                          dest)
        out: dict = {"messages": out_messages, "session_trials": trials,
                     "session_research": research,
                     "session_best": sess_best, "best_trial": run_best,
                     # a batch where EVERY call was skipped is an unproductive
                     # turn; a productive one clears the strike counter
                     "gen_no_tool": (0 if executed_any
                                     else state.get("gen_no_tool", 0) + 1),
                     # asked to submit but evaluated again anyway -> force it
                     "force_official": bool(
                         n_before >= self.max_session_trials
                         and any(c["name"] == self.evaluate_tool_name
                                 for c in last.tool_calls))}
        if sess_best.get("code"):
            out["current_code"] = sess_best["code"]
        return out

    def evaluate_node(self, state: PipelineState) -> dict:
        """OFFICIAL evaluation -- no LLM.

        Re-runs the real evaluator on the submitted design (current_code: the
        last trial's code, or the code block in the designer's final reply)
        and records the result as the next official variant. This is the ONLY
        place best-so-far / iteration are updated. Resets the session budget.
        """
        code = state.get("current_code", "")
        variant_num = state.get("iteration", 0) + 1
        # name the submission so designer and reviewer talk about the SAME code
        # (the session's best trial may not be the designer's latest attempt)
        sess_best = state.get("session_best") or {}
        if not code:
            submitted = "no design"
        elif code == sess_best.get("code"):
            submitted = f"your trial {sess_best.get('trial')}"
        else:
            submitted = "the code block in your final message"

        def _official(c: str, tag: str) -> tuple[dict, float]:
            t0 = time.perf_counter()
            try:
                kwargs = dict(self.evaluate_kwargs)
                kwargs.setdefault("iteration", tag)
                m = self.task.evaluate(c, **kwargs)
            except Exception as exc:
                m = {"ok": False, "error": f"official evaluation raised: {exc}"}
            return m, time.perf_counter() - t0

        if code:
            metrics, dt = _official(code, f"official{variant_num}")
        else:
            metrics, dt = {"ok": False, "error": "no design was submitted"}, 0.0

        # FALLBACK: the designer's chosen code errored -> submit the session's
        # best trial instead (later trial wins ties), so a broken final code
        # block can't waste the whole session
        if (not metrics.get("ok") and sess_best.get("code")
                and sess_best["code"] != code):
            self.task.log(
                f"EVALUATE variant #{variant_num}: submitted code errored -> "
                f"falling back to session best (trial {sess_best.get('trial')})",
                json.dumps(metrics))
            code = sess_best["code"]
            submitted = (f"your trial {sess_best.get('trial')} (fallback: "
                         "the code block you submitted errored)")
            metrics, dt = _official(code, f"official{variant_num}fallback")

        score = metrics.get(self.target_metric, -1.0) if metrics.get("ok") else -1.0
        best = dict(state.get("best") or {})
        # >= so an equal-scoring LATER variant becomes the new best
        if metrics.get("ok") and score >= best.get(self.target_metric, -1.0):
            best = {"code": code, self.target_metric: score,
                    "metrics": metrics, "variant": variant_num}
        self.task.log(
            f"EVALUATE official variant #{variant_num} = {submitted} "
            f"[eval {dt:.2f}s] (best-so-far: variant #{best.get('variant', '?')})",
            f"--- code ---\n{code}\n--- result ---\n{json.dumps(metrics)}")
        return {"last_metrics": {**metrics, "variant": variant_num,
                                 "submitted": submitted},
                "best": best, "iteration": variant_num,
                "session_trials": 0,      # fresh budgets for the next session
                "session_research": 0,
                "session_best": {},       # next session tracks its own best
                "force_official": False,  # ask-then-force cycle starts over
                "gen_no_tool": 0,         # a submission is not a refusal
                "messages": [HumanMessage(content=(
                    f"Official evaluation of your submitted design "
                    f"(variant #{variant_num} = {submitted}): "
                    f"{json.dumps(metrics)}"))]}

    def review_node(self, state: PipelineState) -> dict:
        """Reviewer with its OWN message thread: it remembers every candidate
        it has critiqued and what it already suggested, so feedback does not
        repeat. Only the distilled feedback string enters the design thread."""
        m = state.get("last_metrics", {})
        code = state.get("current_code", "")
        submitted = m.get("submitted", "the submitted design")
        prompt = (f"Official variant #{m.get('variant', '?')} "
                  f"({submitted}) with metrics: {json.dumps(m)}\n"
                  f"Code:\n```python\n{code}\n```\n"
                  f"Give one concrete, actionable improvement (or the fix if it "
                  f"errored). Avoid repeating advice you already gave.")
        # the transcript must show exactly what the reviewer was given
        # (metrics AND code), not just its answer
        self.task.log(f"REVIEW input (variant #{m.get('variant', '?')})", prompt)
        # the seed library rides in the SYSTEM message: the reviewer sees it on
        # every call, but it is never appended to the accumulating review
        # thread, so it is not duplicated round after round.
        review_system = (self.task.system_prompt("review")
                         + "\n\n# Seed library (known-good reference designs)\n"
                         + self._seed_block())
        self._log_system("review", review_system)
        resp, _, dt = self._run_with_tools(
            self.review_llm, self.review_tools,
            [SystemMessage(content=review_system)]
            + state.get("review_messages", [])
            + [HumanMessage(content=prompt)],
        )
        note = _text_of(resp)
        self.task.log(f"REVIEW (round {state.get('iteration', 0)}) [llm {dt:.2f}s]", note)
        # design-thread messages that lead into the NEXT session's generate:
        # the reviewer's feedback, then (chronologically here, NOT in the system
        # prompt) the "build on the current best" reminder.
        feedback_msg = (
            f"Reviewer feedback on variant #{m.get('variant', '?')} ({submitted}) "
            f"-- NOT necessarily your latest attempt:\n{note}")
        design_msgs = [HumanMessage(content=feedback_msg)]
        foundation = self._best_so_far_note(state)
        if foundation:
            design_msgs.append(HumanMessage(content=foundation))
        # log the design-thread hand-off exactly as the NEXT generate will see
        # it, so the transcript shows what actually steered the next session --
        # the reviewer feedback and the "current best, build on it" reminder
        # (previously these were fed to generate but never recorded).
        self.task.log(
            f"NEXT SESSION input (best-so-far: variant "
            f"#{(state.get('best') or {}).get('variant', '?')})",
            feedback_msg + (f"\n\n{foundation}" if foundation else ""))
        return {"review": note,
                # reviewer's own memory: the question + its distilled answer
                # (a fresh AIMessage, so no dangling tool-call pairs persist)
                "review_messages": [HumanMessage(content=prompt),
                                    AIMessage(content=note)],
                "messages": design_msgs}

    def _best_so_far_note(self, state: PipelineState) -> str:
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
            f"({self.target_metric}={best.get(self.target_metric)}, "
            f"gates={bm.get('n_gates')}, depth={bm.get('depth')}) -- it is saved "
            "and cannot be lost. Build on it per your Strategy (refine it to beat "
            "it, or try a fresh direction if it is weak).")

    def deploy_node(self, state: PipelineState) -> dict:
        best = dict(state.get("best") or {})
        iteration = state.get("iteration", 0)

        # SALVAGE: if the best trial of the run still beats the best official
        # variant (e.g. the designer submitted a different design), officially
        # evaluate it now so the best circuit ever tested is never lost.
        bt = state.get("best_trial") or {}
        if bt.get("code") and bt.get("score", -1.0) > best.get(self.target_metric, -1.0):
            iteration += 1
            t0 = time.perf_counter()
            try:
                kwargs = dict(self.evaluate_kwargs)
                kwargs.setdefault("iteration", f"official{iteration}salvage")
                metrics = self.task.evaluate(bt["code"], **kwargs)
            except Exception as exc:
                metrics = {"ok": False,
                           "error": f"official evaluation raised: {exc}"}
            score = (metrics.get(self.target_metric, -1.0)
                     if metrics.get("ok") else -1.0)
            self.task.log(
                f"EVALUATE official variant #{iteration} "
                f"[eval {time.perf_counter() - t0:.2f}s] "
                f"(SALVAGED best trial {bt.get('trial')}, never submitted)",
                f"--- code ---\n{bt['code']}\n--- result ---\n{json.dumps(metrics)}")
            if score > best.get(self.target_metric, -1.0):
                best = {"code": bt["code"], self.target_metric: score,
                        "metrics": metrics, "variant": iteration}

        extra = (f"\nAdditional user instructions for deployment:\n{self.deploy_prompt}"
                 if self.deploy_prompt else "")
        deploy_system = self.task.system_prompt("deploy")
        self._log_system("deploy", deploy_system)
        t0 = time.perf_counter()
        summary_resp = self.deploy_llm.invoke([
            SystemMessage(content=deploy_system),
            HumanMessage(content=(
                f"The search finished after {iteration} official "
                f"variant(s) ({self._eval_count} trial evaluations in total). "
                f"The best artifact was variant #{best.get('variant', '?')}; "
                f"its code and stats:\n{json.dumps(best, indent=2)}\n"
                f"Summarize what was designed (mention which variant won), the "
                f"final stats, and the rationale.{extra}")),
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
                "best": best, "iteration": iteration,   # includes any salvage
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
        # <= max (not <): after the max-th research round we still enter
        # explore_tools once more to run its tools and append the wrap-up
        # warning; the (max+1)-th explorer turn then writes notes and hands off
        if _has_tool_calls(last) and state.get("explore_rounds", 0) <= self.max_explore_rounds:
            return "explore_tools"
        return "generate"          # done researching -> design

    def _route_after_generate(self, state: PipelineState) -> str:
        last = state["messages"][-1]
        if _has_tool_calls(last):
            return "gen_tools"      # research or trial call(s) -> executor
        if isinstance(last, HumanMessage):
            # generate_node injected a corrective nudge -> let it retry,
            # unless it keeps slipping: then submit what we have (or give up)
            if state.get("gen_no_tool", 0) >= self.max_gen_no_tool:
                return ("evaluate" if state.get("session_trials", 0) > 0
                        else "deploy")
            return "generate"
        # clean plain reply = submission (trials always > 0 here: a
        # zero-trial plain reply is always nudged above)
        return "evaluate" if state.get("session_trials", 0) > 0 else "generate"

    def _route_after_gen_tools(self, state: PipelineState) -> str:
        # ALL trial results return to generate -- including the last one, so it
        # can analyze it and choose its submission. Official evaluation is only
        # forced on a designer that keeps calling evaluate after being asked.
        if state.get("force_official"):
            return "evaluate"
        if state.get("gen_no_tool", 0) >= self.max_gen_no_tool:
            # repeated fully-skipped batches (exhausted budgets): stop looping
            return ("evaluate" if state.get("session_trials", 0) > 0
                    else "deploy")
        return "generate"

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
                                {"gen_tools": "gen_tools", "evaluate": "evaluate",
                                 "generate": "generate", "deploy": "deploy"})
        g.add_conditional_edges("gen_tools", self._route_after_gen_tools,
                                {"evaluate": "evaluate", "generate": "generate",
                                 "deploy": "deploy"})
        g.add_conditional_edges("evaluate", self._route_after_evaluate,
                                {"review": "review", "deploy": "deploy"})
        g.add_conditional_edges("review", self._route_after_review,
                                {"generate": "generate", "deploy": "deploy"})
        g.add_edge("deploy", END)
        return g.compile()

    def _dump_histories(self, final: dict) -> dict:
        """Write each message channel to its own readable .txt in the run
        folder. System prompts are not stored in the channels (they are
        rebuilt per call), so each file starts with the LAST system prompt
        used by that channel's agent."""
        channels = {
            "design": ("generate", final.get("messages") or []),
            "explorer": ("explorer", final.get("explore_messages") or []),
            "review": ("review", final.get("review_messages") or []),
        }
        paths = {}
        for name, (role, msgs) in channels.items():
            if not msgs:
                continue
            path = os.path.join(self.task.run_dir, f"history_{name}.txt")
            try:
                with open(path, "w", encoding="utf-8") as f:
                    system = self._logged_systems.get(role)
                    if system:
                        f.write(f"{'=' * 72}\n[system prompt -- final version; "
                                f"earlier versions in transcript.txt]\n"
                                f"{'=' * 72}\n{system}\n")
                    for i, m in enumerate(msgs, 1):
                        role_name = type(m).__name__.replace("Message", "")
                        tool_name = getattr(m, "name", None)
                        header = f"[{i}] {role_name}" + (
                            f" ({tool_name})" if tool_name else "")
                        f.write(f"\n{'=' * 72}\n{header}\n{'=' * 72}\n")
                        text = _text_of(m).strip()
                        if text:
                            f.write(text + "\n")
                        for call in (getattr(m, "tool_calls", None) or []):
                            f.write(f"\n[tool_call] {call['name']}\n")
                            for k, v in (call.get("args") or {}).items():
                                f.write(f"{k}:\n{v}\n")
                paths[name] = path
            except Exception as exc:
                logger_msg = f"failed to write {path}: {exc}"
                self.task.log("HISTORY DUMP ERROR", logger_msg)
        if paths:
            self.task.log("MESSAGE HISTORIES",
                          "\n".join(f"{k}: {v}" for k, v in paths.items()))
        return paths

    # --------------------------------------------------------------- run
    def run(self, extra_instructions: str = "") -> dict:
        design_first = HumanMessage(content=(
            "Design and iteratively improve the artifact. Start by calling "
            "`view_seed_library` to see known-good reference designs you may "
            f"adapt or combine. {extra_instructions}\n"
            f"Test candidates with the `{self.evaluate_tool_name}` tool -- you "
            f"have a budget of {self.max_session_trials} trials per design "
            "session, and each result comes back to you so you can iterate. "
            "When you are happy with a design, reply WITHOUT calling any tool "
            "and include your chosen design in a ```python block (otherwise "
            "your session's best trial is used); it will then be officially "
            "evaluated and reviewed."))

        init_state: dict = {"messages": [design_first], "review_messages": [],
                            "notes": "", "iteration": 0, "best": {},
                            "session_trials": 0, "session_research": 0,
                            "session_best": {},
                            "best_trial": {}, "force_official": False,
                            "gen_no_tool": 0}
        if self.use_explorer:
            init_state["explore_messages"] = [HumanMessage(content=(
                "Research this design problem so the designer can start well "
                f"informed. {extra_instructions}"))]
            init_state["explore_rounds"] = 0

        self.task.log("RUN CONFIG",
                      f"{self.task.describe()}\n"
                      f"explorer={'ON' if self.use_explorer else 'OFF'} "
                      f"notes_tool={'ON' if self.use_notes_tool else 'OFF'} "
                      f"explorer_tools={[t.name for t in self._explorer_toolset()]}\n"
                      f"design_tools={[t.name for t in self._design_tools()]} "
                      f"review_tools={[t.name for t in self.review_tools]}\n"
                      f"max_iters={self.max_iters} (official variants) "
                      f"target={self.target_metric}>={self.target_value} "
                      f"entry_point={self.entry_point}\n"
                      f"max_session_trials={self.max_session_trials} "
                      f"max_session_research={self.max_session_research} "
                      f"max_explore_rounds={self.max_explore_rounds} "
                      f"max_gen_no_tool={self.max_gen_no_tool} "
                      f"max_tool_rounds={self.max_tool_rounds} (review tool calls)")
        run_t0 = time.perf_counter()
        app = self.build_graph()
        final = app.invoke(init_state, {"recursion_limit": 100})
        self.task.log("TOTAL RUNTIME",
                      f"{time.perf_counter() - run_t0:.2f}s over "
                      f"{final.get('iteration', 0)} official variant(s) / "
                      f"{self._eval_count} trial(s)")
        history_paths = self._dump_histories(final)
        return {"history_paths": history_paths,
                "best": final.get("best", {}),
                "summary": final.get("review", ""),
                "notes": final.get("notes", ""),
                "translated": final.get("last_metrics", {}).get("translated"),
                "iterations": final.get("iteration", 0),
                "trials": self._eval_count,
                "log_path": self.task.log_path,
                "notes_path": self.notes_path}
