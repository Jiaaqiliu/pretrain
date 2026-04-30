"""ComputeTarget implementations — where a DDP stage actually runs.

Each target conforms to the ``compute_target.ComputeTarget`` Protocol so the
``ElasticScheduler`` can route stages uniformly:

- ``local`` — spawn torchrun on the host GPUs (file-locked pool).
- ``k8s``   — submit a ``batch/v1`` Job to the shared cluster.
"""
