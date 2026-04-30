"""K8s-specific assets for the elastic backend.

- ``job_manifest`` builds the ``batch/v1`` Job body the K8sComputeTarget submits.
- ``image/`` contains Docker build artifacts (not importable Python).
- ``smoke/`` contains host-side drivers + manual kubectl manifests.
"""
