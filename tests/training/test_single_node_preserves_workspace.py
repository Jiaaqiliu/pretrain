"""PR7 acceptance: candidate workspace survives a failed trial."""

from __future__ import annotations

from pathlib import Path

from agent_evolve.backends.tinkerlite.single_node import SingleNodeTinkerLiteBackend
from agent_evolve.training.algorithms.mcgs import MCGSSearch
from agent_evolve.training.loop import TrainingEvolutionLoop
from agent_evolve.training.workspace import TrainingWorkspace

from .fakes import FakeBenchmark


def test_failure_preserves_candidate_dir(minimal_workspace: Path, tmp_path: Path) -> None:
    # Remove pipeline.yaml AFTER loading so MCGS can still fork (copy has it),
    # then we break the pipeline in the child workspace to trigger train_failed.
    ws = TrainingWorkspace.load(minimal_workspace)
    backend = SingleNodeTinkerLiteBackend(mock=True)

    # Hook: delete pipeline inside the forked workspace just before training.
    original = backend._run_pipeline

    def _break(workspace, pipeline, budget):  # noqa: ARG001
        raise RuntimeError("simulated stage crash")

    backend._run_pipeline = _break  # type: ignore[method-assign]

    algo = MCGSSearch()
    loop = TrainingEvolutionLoop(
        workspace=ws,
        benchmark=FakeBenchmark(),
        algorithm=algo,
        backend=backend,
        work_dir=tmp_path / "work",
    )
    loop.run(cycles=1)

    # Candidate workspace dir should still be on disk.
    node_dirs = list((tmp_path / "work" / "nodes").iterdir())
    assert node_dirs, "expected at least one candidate node dir"
    for node_dir in node_dirs:
        assert (node_dir / "workspace").is_dir()

    backend._run_pipeline = original  # type: ignore[method-assign]
