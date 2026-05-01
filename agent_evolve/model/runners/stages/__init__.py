"""One module per pipeline ``stage.type``.

Every module here exposes a ``run_<type>_stage(workspace, stage, ...) ->
(CheckpointRef | Path, stats_dict)`` entrypoint that the backend's pipeline
dispatcher invokes. Stage-support utilities (dataset rendering, adapter
packaging) live in ``runners/helpers/``, not here.
"""
