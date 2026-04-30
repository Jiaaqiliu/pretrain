"""K8s-first elastic TinkerLite backend.

Exposes ``K8sTinkerLiteBackend`` (key ``k8s_h200`` in the training registry).
The backend is "k8s-first, local-fallback": submits to a shared k8s cluster
by default, queues on cluster pressure with a configurable timeout, and
falls back to the local machine (if enabled) when cluster capacity is
exhausted or the queue wait exceeds the threshold.

Sibling backend ``SingleNodeTinkerLiteBackend`` (``h200_single_node``)
remains the zero-infrastructure path for single deep trials.
"""

from .backend import K8sTinkerLiteBackend
from .scheduler import FanoutCapacity, StageHandle

__all__ = ["K8sTinkerLiteBackend", "FanoutCapacity", "StageHandle"]
