"""PipelineState: the LangGraph state for AgentPipeline.

Split out of AgentPipeline.py for readability; see that module's docstring for
how the three message channels and the trial/variant bookkeeping interact.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


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
