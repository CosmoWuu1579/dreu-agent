"""Put the parent ``dreu-agent/`` directory on ``sys.path``.

These toy-problem scripts live one level below the framework modules they
depend on (``AgentPipeline``, ``DiscoveryTask``, ``db.*``). Importing this
module first makes those parent-level imports resolve no matter what directory
the script is launched from. Import it for its side effect, before any
parent-level import::

    import _bootstrap  # noqa: F401  (adds parent dir to sys.path)
    from AgentPipeline import AgentPipeline
"""

import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
