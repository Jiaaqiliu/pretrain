"""Interactive conversational mode for AutoPilot.

Provides a REPL-style interface where users can:
- Configure training in natural language
- Ask about training progress
- Make adjustments on the fly
- Get explanations of agent decisions
"""

from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from autopilot.agent.data_manager import DataManagerAgent
from autopilot.agent.natural_language import NaturalLanguageInterface
from autopilot.agent.orchestrator import AgentConfig, AutoPilotAgent, AutonomyLevel
from autopilot.backends.base import ComputeBackend
from autopilot.utils.logging import get_logger

log = get_logger("ui.interactive")

console = Console()

WELCOME_MESSAGE = """
# Welcome to AutoPilot Interactive Mode

I'm your autonomous LLM training agent. You can tell me what you want to train
in natural language, and I'll handle the rest.

**Examples:**
- "Train a 7B model on web and code data with 8 nodes"
- "I have 500B tokens of web data and 200B of code data"
- "Use H100 GPUs, target loss 2.5"
- "Show me the current progress"
- "Stop the worst experiment"
- "What's the best learning rate so far?"

Type `help` for commands, `quit` to exit.
"""


class InteractiveSession:
    """Interactive REPL for conversational training management."""

    def __init__(self, backend: ComputeBackend, config: Optional[AgentConfig] = None):
        self._backend = backend
        self._config = config or AgentConfig(autonomy_level=AutonomyLevel.SEMI)
        self._nl = NaturalLanguageInterface()
        self._data_manager = DataManagerAgent()
        self._agent: Optional[AutoPilotAgent] = None
        self._running = True

    def run(self) -> None:
        """Start the interactive session."""
        console.print(Markdown(WELCOME_MESSAGE))
        console.print()

        while self._running:
            try:
                user_input = Prompt.ask("[bold cyan]autopilot[/]")
                if not user_input.strip():
                    continue
                self._handle_input(user_input.strip())
            except KeyboardInterrupt:
                console.print("\n[dim]Use 'quit' to exit.[/]")
            except EOFError:
                break

        console.print("[dim]Goodbye![/]")

    def _handle_input(self, text: str) -> None:
        # Handle built-in commands
        lower = text.lower()
        if lower in ("quit", "exit", "q"):
            self._running = False
            return
        if lower == "help":
            self._show_help()
            return
        if lower == "status":
            self._show_status()
            return
        if lower == "plan":
            self._show_plan()
            return
        if lower == "experiments":
            self._show_experiments()
            return
        if lower == "decisions":
            self._show_decisions()
            return
        if lower == "report":
            self._show_report()
            return

        # Parse natural language
        intent = self._nl.parse(text)

        if intent.action == "train":
            self._handle_train(text, intent)
        elif intent.action == "data":
            self._handle_data(text)
        elif intent.action == "status":
            self._show_status()
        elif intent.action == "stop":
            self._handle_stop(text)
        elif intent.action == "deploy":
            self._handle_deploy()
        elif intent.action == "adjust":
            self._handle_adjust(text, intent)
        elif intent.action == "configure":
            self._handle_configure(text, intent)
        else:
            # Default: try to be helpful
            console.print(
                f"[dim]I understood your intent as '{intent.action}'. "
                f"Let me help with that.[/]"
            )
            self._handle_generic(text, intent)

    def _handle_train(self, text: str, intent) -> None:
        """Handle training requests."""
        target = self._nl.to_training_target(intent)

        console.print(Panel.fit(
            f"[bold]Training Plan[/]\n"
            f"Model: {target.model_size.value} "
            f"({target.model_size.to_scale().num_params_approx/1e9:.1f}B params)\n"
            f"Phase: {target.phase.value}\n"
            f"Data: {', '.join(target.data_domains) if target.data_domains else 'auto'}\n"
            f"Target loss: {target.target_loss or 'none specified'}\n"
            f"Compute: {target.compute_budget.total_gpus if target.compute_budget else 8} GPUs",
            title="Proposed Configuration",
        ))

        confirm = Prompt.ask("Proceed?", choices=["yes", "no", "modify"], default="yes")
        if confirm == "no":
            console.print("[yellow]Cancelled.[/]")
            return
        elif confirm == "modify":
            console.print("[dim]Tell me what to change.[/]")
            return

        # Update config and create agent
        self._config.model_size = target.model_size
        self._config.data_domains = target.data_domains
        if target.target_loss:
            self._config.target_loss = target.target_loss
        if target.compute_budget:
            self._config.compute_budget = target.compute_budget

        self._agent = AutoPilotAgent(config=self._config, backend=self._backend)
        plan = self._agent.start()

        console.print(f"\n[green]Campaign started![/] {len(plan.phases)} phases planned.")
        console.print("[dim]The agent is now running. Type 'status' to check progress.[/]")

    def _handle_data(self, text: str) -> None:
        """Handle data-related requests."""
        spec = self._nl.parse_data_spec(text)

        # Register available data
        for domain in spec.available_domains:
            self._data_manager.register_data(
                name=domain.name, path=domain.path, token_count=domain.token_count
            )
            console.print(
                f"  [green]Registered:[/] {domain.name} "
                f"({domain.token_count/1e9:.0f}B tokens)"
            )

        # Handle mixture preferences
        if spec.mixture_preferences:
            console.print("\n[bold]Mixture preferences:[/]")
            for domain, weight in spec.mixture_preferences.items():
                console.print(f"  {domain}: {weight*100:.0f}%")

        # Handle missing data
        if spec.missing_domains:
            console.print(f"\n[yellow]Missing domains:[/] {', '.join(spec.missing_domains)}")
            suggestions = self._data_manager.suggest_sources(spec.missing_domains)
            if suggestions:
                console.print("\n[bold]Suggested sources to fill gaps:[/]")
                table = Table()
                table.add_column("Name", style="cyan")
                table.add_column("Tokens")
                table.add_column("Quality")
                table.add_column("License")
                for s in suggestions[:5]:
                    table.add_row(
                        s.name,
                        f"{s.estimated_tokens/1e9:.0f}B",
                        s.quality_tier,
                        s.license,
                    )
                console.print(table)

        # Show current registered data
        all_domains = self._data_manager.available_domains
        if all_domains:
            console.print(f"\n[bold]Total registered:[/] {len(all_domains)} domains, "
                         f"{sum(d.token_count for d in all_domains)/1e9:.0f}B tokens")

    def _handle_stop(self, text: str) -> None:
        """Handle stop requests."""
        if self._agent is None:
            console.print("[yellow]No active campaign.[/]")
            return

        active = self._agent.active_experiments
        if not active:
            console.print("[dim]No running experiments.[/]")
            return

        console.print(f"Active experiments: {len(active)}")
        confirm = Prompt.ask("Stop all?", choices=["yes", "no", "worst"], default="worst")

        if confirm == "yes":
            self._agent.stop()
            console.print("[red]All experiments stopped.[/]")
        elif confirm == "worst":
            console.print("[dim]Would stop the worst-performing experiment.[/]")

    def _handle_deploy(self) -> None:
        """Handle deployment/environment setup."""
        from autopilot.agent.environment import EnvironmentAgent

        env_agent = EnvironmentAgent()
        report = env_agent.discover()

        console.print(Panel.fit(
            f"[bold]Environment Report[/]\n"
            f"Host: {report.hostname}\n"
            f"OS: {report.os_info}\n"
            f"Python: {report.python_version}\n"
            f"CUDA: {report.cuda_version or 'not found'}\n"
            f"PyTorch: {report.torch_version or 'not installed'}\n"
            f"GPUs: {len(report.gpus)}\n"
            f"Cluster: {report.cluster.cluster_type if report.cluster else 'none'}",
            title="Environment Discovery",
        ))

        if report.issues:
            console.print("\n[red]Issues:[/]")
            for issue in report.issues:
                console.print(f"  - {issue}")

        if report.recommendations:
            console.print("\n[yellow]Recommendations:[/]")
            for rec in report.recommendations:
                console.print(f"  - {rec}")

        setup_cmds = env_agent.setup_environment(report)
        if setup_cmds:
            console.print("\n[bold]Setup commands:[/]")
            for cmd in setup_cmds:
                console.print(f"  [dim]$ {cmd}[/]")

    def _handle_adjust(self, text: str, intent) -> None:
        """Handle adjustment requests."""
        if intent.parameters:
            console.print(f"[bold]Adjustments:[/] {intent.parameters}")
        else:
            console.print("[dim]What would you like to adjust? (e.g., 'increase learning rate')[/]")

    def _handle_configure(self, text: str, intent) -> None:
        """Handle configuration changes."""
        if intent.compute_config:
            console.print(f"[bold]Compute config updated:[/] {intent.compute_config}")
        if intent.data_config:
            console.print(f"[bold]Data config updated:[/] {intent.data_config}")
        if intent.model_size:
            console.print(f"[bold]Model size:[/] {intent.model_size.value}")

    def _handle_generic(self, text: str, intent) -> None:
        """Handle generic queries that don't fit other categories."""
        console.print(
            "[dim]I can help with: training, data management, status checks, "
            "stopping experiments, and environment setup. "
            "Try rephrasing or type 'help'.[/]"
        )

    def _show_help(self) -> None:
        console.print(Markdown("""
## Commands

| Command | Description |
|---------|-------------|
| `status` | Show campaign status |
| `plan` | Show current training plan |
| `experiments` | List all experiments |
| `decisions` | Show decision history |
| `report` | Generate analysis report |
| `help` | Show this help |
| `quit` | Exit |

## Natural Language Examples

- **Start training**: "Train a 7B model on web and code data"
- **Configure data**: "I have 500B tokens of web data"
- **Check progress**: "How is training going?"
- **Adjust**: "Increase learning rate to 5e-4"
- **Stop**: "Stop the failing experiments"
- **Deploy**: "Set up the environment for training"
"""))

    def _show_status(self) -> None:
        if self._agent is None:
            console.print("[dim]No active campaign. Start one with a training command.[/]")
            return

        dashboard = self._agent.get_dashboard_data()
        console.print(Panel.fit(
            f"[bold]Status:[/] {dashboard['status']}\n"
            f"Phase: {dashboard['current_phase']}\n"
            f"Active experiments: {len(dashboard['active_experiments'])}\n"
            f"HPO trials completed: {dashboard['hpo_completed']}\n"
            f"Best params: {dashboard['hpo_best']}",
            title="Campaign Status",
        ))

        if dashboard['rankings']:
            console.print("\n[bold]Rankings:[/]")
            for eid, loss in dashboard['rankings'][:5]:
                console.print(f"  {eid[:12]}: loss={loss:.4f}")

    def _show_plan(self) -> None:
        if self._agent is None or self._agent.plan is None:
            console.print("[dim]No plan yet.[/]")
            return

        plan = self._agent.plan
        table = Table(title=f"Training Plan: {plan.name}")
        table.add_column("#")
        table.add_column("Phase")
        table.add_column("Experiments")
        table.add_column("GPU-hrs")
        for i, phase in enumerate(plan.phases):
            marker = "→" if i == self._agent._current_phase_idx else " "
            table.add_row(
                f"{marker}{i+1}",
                phase.name,
                str(phase.n_experiments),
                f"{phase.estimated_gpu_hours:.1f}",
            )
        console.print(table)

    def _show_experiments(self) -> None:
        if self._agent is None:
            console.print("[dim]No experiments.[/]")
            return
        console.print(f"Active: {self._agent.active_experiments}")

    def _show_decisions(self) -> None:
        console.print("[dim]Decision history requires active campaign.[/]")

    def _show_report(self) -> None:
        if self._agent is None:
            console.print("[dim]No data for report.[/]")
            return
        report = self._agent.get_report()
        console.print(f"\n[bold]{report.summary}[/]")
        for rec in report.recommendations:
            console.print(f"  - {rec}")
