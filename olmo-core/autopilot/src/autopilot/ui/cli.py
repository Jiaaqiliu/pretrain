"""AutoPilot CLI — command-line interface for the autonomous training agent."""

from __future__ import annotations

from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from autopilot.agent.orchestrator import AgentConfig, AutoPilotAgent, AutonomyLevel
from autopilot.backends.beaker import BeakerBackend
from autopilot.backends.slurm import SlurmBackend
from autopilot.experiment.config_builder import ComputeBudget, ModelSize
from autopilot.utils.logging import setup_logging

console = Console()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool):
    """AutoPilot: Autonomous LLM Training Agent"""
    import logging

    setup_logging(level=logging.DEBUG if verbose else logging.INFO)


@main.command()
@click.option("--model-size", type=click.Choice(["tiny", "small", "medium", "large", "xl", "xxl"]), default="medium")
@click.option("--target-loss", type=float, default=None, help="Target validation loss")
@click.option("--num-nodes", type=int, default=1, help="Number of compute nodes")
@click.option("--gpus-per-node", type=int, default=8, help="GPUs per node")
@click.option("--gpu-type", type=str, default="A100-80GB", help="GPU type")
@click.option("--backend", type=click.Choice(["beaker", "slurm"]), default="beaker")
@click.option("--autonomy", type=click.Choice(["full", "semi", "advisory"]), default="semi")
@click.option("--data-domains", type=str, multiple=True, help="Data domains to use")
@click.option("--state-dir", type=str, default="./autopilot_state", help="State persistence directory")
@click.option("--include-sft", is_flag=True, help="Include SFT phase after pretraining")
@click.option("--max-parallel", type=int, default=8, help="Maximum parallel experiments")
def train(
    model_size: str,
    target_loss: Optional[float],
    num_nodes: int,
    gpus_per_node: int,
    gpu_type: str,
    backend: str,
    autonomy: str,
    data_domains: tuple,
    state_dir: str,
    include_sft: bool,
    max_parallel: int,
):
    """Start an autonomous training campaign."""
    console.print(Panel.fit(
        "[bold blue]AutoPilot[/] — Autonomous LLM Training Agent",
        subtitle="Starting new campaign",
    ))

    # Create backend
    compute_backend = _create_backend(backend)

    # Build config
    config = AgentConfig(
        store_dir=state_dir,
        autonomy_level=AutonomyLevel(autonomy),
        model_size=ModelSize(model_size),
        target_loss=target_loss,
        compute_budget=ComputeBudget(
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            gpu_type=gpu_type,
        ),
        data_domains=list(data_domains),
        include_sft=include_sft,
        max_parallel_experiments=max_parallel,
    )

    # Show plan summary
    agent = AutoPilotAgent(config=config, backend=compute_backend)
    plan = agent.start()

    _display_plan(plan)

    # Confirm before running
    if autonomy != "full":
        if not click.confirm("\nProceed with this plan?"):
            console.print("[yellow]Aborted.[/]")
            return

    # Run the agent
    console.print("\n[green]Starting autonomous training loop...[/]")
    console.print(f"Autonomy level: [bold]{autonomy}[/]")
    console.print(f"Poll interval: {config.poll_interval_seconds}s")
    console.print("Press Ctrl+C to pause.\n")

    agent.run_loop()


@main.command()
@click.option("--state-dir", type=str, default="./autopilot_state")
def status(state_dir: str):
    """Show current campaign status."""
    from autopilot.utils.persistence import StateStore

    store = StateStore(state_dir)
    current_status = store.get_state("status", "unknown")
    current_phase = store.get_state("current_phase", 0)

    console.print(Panel.fit(f"[bold]Status:[/] {current_status}  |  Phase: {current_phase}"))

    # Show experiments
    experiments = store.list_experiments()
    if experiments:
        table = Table(title="Experiments")
        table.add_column("ID", style="cyan")
        table.add_column("Name")
        table.add_column("Status", style="green")
        table.add_column("Updated")

        for exp in experiments[:20]:
            import datetime

            updated = datetime.datetime.fromtimestamp(exp.updated_at).strftime("%H:%M:%S")
            status_style = {
                "running": "green",
                "completed": "blue",
                "failed": "red",
                "stopped": "yellow",
            }.get(exp.status, "white")
            table.add_row(
                exp.experiment_id[:12],
                exp.name,
                f"[{status_style}]{exp.status}[/]",
                updated,
            )

        console.print(table)
    else:
        console.print("[dim]No experiments recorded.[/]")


@main.command()
@click.option("--state-dir", type=str, default="./autopilot_state")
@click.option("--limit", type=int, default=20)
def decisions(state_dir: str, limit: int):
    """Show decision history."""
    from autopilot.utils.persistence import StateStore

    store = StateStore(state_dir)
    records = store.get_decisions(limit=limit)

    if not records:
        console.print("[dim]No decisions recorded.[/]")
        return

    table = Table(title="Decision History")
    table.add_column("Time", style="dim")
    table.add_column("Type", style="cyan")
    table.add_column("Experiment")
    table.add_column("Reasoning")

    for rec in records:
        import datetime

        ts = datetime.datetime.fromtimestamp(rec.timestamp).strftime("%m-%d %H:%M")
        table.add_row(ts, rec.decision_type, rec.experiment_id[:12], rec.reasoning[:60])

    console.print(table)


@main.command()
@click.option("--state-dir", type=str, default="./autopilot_state")
def report(state_dir: str):
    """Generate analysis report for the current campaign."""
    console.print("[bold]Generating analysis report...[/]")
    console.print("[dim]Report generation requires active experiments with metrics.[/]")


@main.command()
@click.option("--backend", type=click.Choice(["beaker", "slurm"]), default="beaker")
def resources(backend: str):
    """Show available compute resources."""
    compute_backend = _create_backend(backend)
    res = compute_backend.get_available_resources()
    console.print(Panel.fit(f"[bold]Available Resources ({backend})[/]"))
    for key, value in res.items():
        console.print(f"  {key}: {value}")


@main.command()
@click.option("--state-dir", type=str, default="./autopilot_state")
@click.option("--backend", type=click.Choice(["beaker", "slurm"]), default="beaker")
def resume(state_dir: str, backend: str):
    """Resume a paused campaign."""
    from autopilot.utils.persistence import StateStore

    store = StateStore(state_dir)
    current_status = store.get_state("status")

    if current_status != "paused":
        console.print(f"[yellow]Campaign is not paused (status: {current_status})[/]")
        return

    console.print("[green]Resuming campaign...[/]")
    # Would reconstruct agent state and continue


@main.command()
@click.option("--backend", type=click.Choice(["beaker", "slurm"]), default="beaker")
@click.option("--state-dir", type=str, default="./autopilot_state")
def interactive(backend: str, state_dir: str):
    """Start interactive conversational mode."""
    from autopilot.ui.interactive import InteractiveSession

    compute_backend = _create_backend(backend)
    config = AgentConfig(store_dir=state_dir, autonomy_level=AutonomyLevel.SEMI)
    session = InteractiveSession(backend=compute_backend, config=config)
    session.run()


@main.command()
@click.option("--backend", type=click.Choice(["beaker", "slurm"]), default="beaker")
def discover(backend: str):
    """Discover and validate the training environment."""
    from autopilot.agent.environment import EnvironmentAgent

    env = EnvironmentAgent()
    report = env.discover()

    console.print(Panel.fit(
        f"Host: {report.hostname}\n"
        f"OS: {report.os_info}\n"
        f"Python: {report.python_version}\n"
        f"CUDA: {report.cuda_version or '[red]not found[/]'}\n"
        f"PyTorch: {report.torch_version or '[red]not installed[/]'}\n"
        f"GPUs: {len(report.gpus)}\n"
        f"Cluster: {report.cluster.cluster_type if report.cluster else 'none'}",
        title="[bold]Environment Discovery[/]",
    ))

    if report.issues:
        console.print("\n[red bold]Issues:[/]")
        for issue in report.issues:
            console.print(f"  [red]•[/] {issue}")

    if report.recommendations:
        console.print("\n[yellow bold]Recommendations:[/]")
        for rec in report.recommendations:
            console.print(f"  [yellow]•[/] {rec}")

    tools_table = Table(title="Available Tools")
    tools_table.add_column("Tool")
    tools_table.add_column("Status")
    for tool, available in report.available_tools.items():
        status = "[green]✓[/]" if available else "[red]✗[/]"
        tools_table.add_row(tool, status)
    console.print(tools_table)


@main.command()
@click.argument("domains", nargs=-1)
@click.option("--scan-path", type=str, default=None, help="Scan directory for data")
def data(domains: tuple, scan_path: str):
    """Manage training data sources."""
    from autopilot.agent.data_manager import DataManagerAgent

    dm = DataManagerAgent()

    if scan_path:
        console.print(f"[bold]Scanning {scan_path} for data...[/]")
        discovered = dm.discover_local_data(scan_path)
        for d in discovered:
            console.print(f"  [green]Found:[/] {d.name} ({d.token_count/1e9:.1f}B tokens)")
        if not discovered:
            console.print("[dim]No data files found.[/]")
        return

    if domains:
        console.print(f"[bold]Suggesting data sources for: {', '.join(domains)}[/]")
        suggestions = dm.suggest_sources(list(domains))
        table = Table(title="Recommended Data Sources")
        table.add_column("Name", style="cyan")
        table.add_column("Tokens")
        table.add_column("Quality")
        table.add_column("Domains")
        table.add_column("License")
        for s in suggestions:
            table.add_row(
                s.name,
                f"{s.estimated_tokens/1e9:.0f}B",
                s.quality_tier,
                ", ".join(s.domains),
                s.license,
            )
        console.print(table)
    else:
        console.print("[dim]Usage: autopilot data web code math[/]")
        console.print("[dim]       autopilot data --scan-path /path/to/data[/]")


def _create_backend(backend_name: str):
    if backend_name == "beaker":
        return BeakerBackend()
    elif backend_name == "slurm":
        return SlurmBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


def _display_plan(plan):
    """Display the training plan in a formatted table."""
    table = Table(title=f"Training Plan: {plan.name}")
    table.add_column("#", style="dim")
    table.add_column("Phase", style="bold")
    table.add_column("Type")
    table.add_column("Experiments", justify="right")
    table.add_column("Est. GPU-hrs", justify="right")
    table.add_column("Description")

    for i, phase in enumerate(plan.phases):
        table.add_row(
            str(i + 1),
            phase.name,
            phase.phase_type,
            str(phase.n_experiments),
            f"{phase.estimated_gpu_hours:.1f}",
            phase.description[:50] + "..." if len(phase.description) > 50 else phase.description,
        )

    console.print(table)
    console.print(
        f"\n[bold]Total estimated cost:[/] {plan.total_estimated_gpu_hours:.0f} GPU-hours"
    )


if __name__ == "__main__":
    main()
