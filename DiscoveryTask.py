"""
DiscoveryTask: a domain-agnostic description of an "algorithm discovery" problem.

An AgentPipeline consumes a DiscoveryTask. To target a new domain -- quantum
feature maps, VQE ansatze, classical algorithms, prompt optimization, ... -- you
build a new DiscoveryTask and the pipeline graph never changes.

Configure by composition (pass dicts / callables to __init__) OR by subclassing
(override the methods). Every field has a sensible default, so partial configs
work. The six capabilities the pipeline relies on:

  system_prompt(agent_type)  -> the prompt for a given agent role (generate / review / deploy / ...)
  seeds(seed_type)           -> a dict of named seed artifacts (kernels / feature maps / algorithms / ...)
  evaluate(code, *args, **kw)-> metrics dict (optional extra args for future evaluators)
  knowledge_sources()        -> RAG sources; retrieve(query, k) fetches chunks
  documentation_source()     -> API/language docs for the target language/library
  log(section, text)         -> print + append everything relevant to the task
"""

from __future__ import annotations

import os
import json
import datetime
from typing import Any, Callable, Optional


class DiscoveryTask:
    def __init__(
        self,
        name: str,
        *,
        system_prompts: Optional[dict[str, str]] = None,
        seeds: Optional[dict[str, dict[str, str]]] = None,
        evaluate_fn: Optional[Callable[..., dict]] = None,
        knowledge_sources: Optional[list[Any]] = None,
        documentation_source: Optional[Any] = None,
        retriever: Optional[Callable[[str, int], list[str]]] = None,
        doc_retriever: Optional[Callable[[str, int], list[str]]] = None,
        log_dir: str = "runs",
    ):
        self.name = name
        self._system_prompts = system_prompts or {}
        self._seeds = seeds or {}
        self._evaluate_fn = evaluate_fn
        self._knowledge_sources = knowledge_sources or []
        self._documentation_source = documentation_source
        self._retriever = retriever            # RAG over knowledge_sources (papers)
        self._doc_retriever = doc_retriever    # RAG over documentation (API docs)
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{name}_{stamp}.txt")

    # --- 1. system prompt, keyed by which agent role is asking ---------------
    def system_prompt(self, agent_type: str = "generate", **fmt) -> str:
        """Return the system prompt for `agent_type`, optionally .format()-ed."""
        template = self._system_prompts.get(agent_type)
        if template is None:
            raise KeyError(f"no system prompt for agent_type={agent_type!r}; "
                           f"available: {list(self._system_prompts)}")
        return template.format(**fmt) if fmt else template

    def agent_types(self) -> list[str]:
        return list(self._system_prompts)

    # --- 2. seeds, grouped by type -------------------------------------------
    def seeds(self, seed_type: Optional[str] = None) -> dict:
        """Named seed artifacts for `seed_type`; all seeds if seed_type is None."""
        if seed_type is None:
            merged: dict[str, str] = {}
            for group in self._seeds.values():
                merged.update(group)
            return merged
        return self._seeds.get(seed_type, {})

    def seed_types(self) -> list[str]:
        return list(self._seeds)

    # --- 3. evaluate: code + optional future args ----------------------------
    def evaluate(self, code: str, *args, **kwargs) -> dict:
        """Score a candidate artifact. Extra *args/**kwargs are forwarded to the
        evaluator, so future evaluators can require more inputs without changing
        this signature."""
        if self._evaluate_fn is None:
            raise NotImplementedError(f"task {self.name!r} has no evaluate_fn")
        return self._evaluate_fn(code, *args, **kwargs)

    # --- 4. knowledge sources + RAG hook (papers) ----------------------------
    def knowledge_sources(self) -> list[Any]:
        return self._knowledge_sources

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Top-k relevant chunks from knowledge sources (empty if no retriever)."""
        return self._retriever(query, k) if self._retriever else []

    # --- 5. documentation source (language/library API docs) -----------------
    def documentation_source(self) -> Any:
        return self._documentation_source

    def retrieve_docs(self, query: str, k: int = 3) -> list[str]:
        """Top-k relevant documentation chunks (empty if no doc retriever)."""
        return self._doc_retriever(query, k) if self._doc_retriever else []

    # --- 6. logging ----------------------------------------------------------
    def log(self, section: str, text: str = "") -> None:
        block = f"\n{'=' * 72}\n[{self.name}] {section}\n{'=' * 72}\n{text}\n"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(block)
        print(f"[{self.name}] {section}")

    def describe(self) -> str:
        return json.dumps({
            "name": self.name,
            "agent_types": self.agent_types(),
            "seed_types": self.seed_types(),
            "n_knowledge_sources": len(self._knowledge_sources),
            "documentation_source": str(self._documentation_source),
            "has_evaluate": self._evaluate_fn is not None,
            "has_retriever": self._retriever is not None,
        }, indent=2)
