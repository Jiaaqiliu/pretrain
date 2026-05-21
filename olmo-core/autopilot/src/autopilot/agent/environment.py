"""Environment discovery and deployment agent.

Autonomously explores the production environment to:
- Detect available compute resources (GPU type, count, clusters)
- Configure networking for distributed training
- Set up dependencies and containers
- Validate the environment before launching training
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from autopilot.utils.logging import get_logger

log = get_logger("agent.environment")


@dataclass
class GPUInfo:
    """Information about a GPU device."""

    index: int
    name: str
    memory_total_gb: float
    compute_capability: str
    driver_version: str


@dataclass
class ClusterInfo:
    """Information about a compute cluster."""

    cluster_type: str  # "slurm", "beaker", "kubernetes", "local"
    total_nodes: int
    gpus_per_node: int
    gpu_type: str
    interconnect: str  # "infiniband", "ethernet", "nvlink"
    storage_paths: Dict[str, str] = field(default_factory=dict)
    available_partitions: List[str] = field(default_factory=list)


@dataclass
class EnvironmentReport:
    """Complete report of the discovered environment."""

    hostname: str
    os_info: str
    python_version: str
    cuda_version: Optional[str]
    torch_version: Optional[str]
    gpus: List[GPUInfo]
    cluster: Optional[ClusterInfo]
    available_tools: Dict[str, bool]  # tool_name -> installed
    storage: Dict[str, Dict[str, Any]]  # path -> {total, free, type}
    network: Dict[str, str]
    issues: List[str]  # any detected problems
    recommendations: List[str]  # suggested fixes


class EnvironmentAgent:
    """Discovers and configures the training environment.

    Workflow:
    1. Detect hardware (GPUs, networking)
    2. Identify cluster type (SLURM, K8s, Beaker, local)
    3. Check software dependencies
    4. Validate storage and network
    5. Generate deployment recommendations
    """

    def discover(self) -> EnvironmentReport:
        """Run full environment discovery."""
        log.info("Discovering environment...")

        report = EnvironmentReport(
            hostname=platform.node(),
            os_info=f"{platform.system()} {platform.release()}",
            python_version=platform.python_version(),
            cuda_version=self._get_cuda_version(),
            torch_version=self._get_torch_version(),
            gpus=self._discover_gpus(),
            cluster=self._discover_cluster(),
            available_tools=self._check_tools(),
            storage=self._check_storage(),
            network=self._check_network(),
            issues=[],
            recommendations=[],
        )

        # Analyze and generate recommendations
        report.issues = self._identify_issues(report)
        report.recommendations = self._generate_recommendations(report)

        log.info(
            f"Environment: {report.hostname}, "
            f"{len(report.gpus)} GPUs, "
            f"cluster={report.cluster.cluster_type if report.cluster else 'none'}"
        )
        return report

    def validate_for_training(
        self, model_size_params: float, num_nodes: int = 1
    ) -> Dict[str, Any]:
        """Validate if the environment can support the planned training."""
        report = self.discover()
        validation = {
            "ready": True,
            "issues": [],
            "warnings": [],
        }

        # Check GPU availability
        if not report.gpus:
            validation["ready"] = False
            validation["issues"].append("No GPUs detected")
        else:
            total_gpu_memory = sum(g.memory_total_gb for g in report.gpus)
            # Rough estimate: model needs ~2x params in bytes for bf16 + optimizer states
            required_memory_gb = model_size_params * 2 / 1e9 * 4  # params * bytes * (model + optim)
            if total_gpu_memory < required_memory_gb / num_nodes:
                validation["warnings"].append(
                    f"GPU memory may be insufficient: "
                    f"{total_gpu_memory:.0f}GB available vs ~{required_memory_gb:.0f}GB needed. "
                    f"FSDP/TP will be required."
                )

        # Check CUDA
        if not report.cuda_version:
            validation["ready"] = False
            validation["issues"].append("CUDA not detected")

        # Check torch
        if not report.torch_version:
            validation["ready"] = False
            validation["issues"].append("PyTorch not installed")

        # Check cluster for multi-node
        if num_nodes > 1 and not report.cluster:
            validation["ready"] = False
            validation["issues"].append(
                "Multi-node training requested but no cluster manager detected"
            )

        # Check NCCL
        if not report.available_tools.get("nccl", False):
            validation["warnings"].append("NCCL not detected; distributed training may fail")

        return validation

    def setup_environment(self, report: Optional[EnvironmentReport] = None) -> List[str]:
        """Generate setup commands to prepare the environment for training."""
        if report is None:
            report = self.discover()

        commands = []

        # Install missing Python dependencies
        if not report.torch_version:
            commands.append("pip install torch --index-url https://download.pytorch.org/whl/cu121")

        if not report.available_tools.get("flash_attn"):
            commands.append("pip install flash-attn --no-build-isolation")

        # Install OLMo-core
        commands.append("pip install -e '.[all]'")

        # Install autopilot
        commands.append("pip install -e './autopilot[all]'")

        # Set up environment variables for distributed training
        if report.cluster and report.cluster.cluster_type == "slurm":
            commands.extend([
                "export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)",
                "export MASTER_PORT=29500",
                "export WORLD_SIZE=$SLURM_NTASKS",
                "export RANK=$SLURM_PROCID",
                "export LOCAL_RANK=$SLURM_LOCALID",
            ])

        return commands

    def _discover_gpus(self) -> List[GPUInfo]:
        """Discover available GPUs via nvidia-smi."""
        gpus = []
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,compute_cap,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        gpus.append(
                            GPUInfo(
                                index=int(parts[0]),
                                name=parts[1],
                                memory_total_gb=float(parts[2]) / 1024,
                                compute_capability=parts[3],
                                driver_version=parts[4],
                            )
                        )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return gpus

    def _discover_cluster(self) -> Optional[ClusterInfo]:
        """Detect the cluster management system."""
        # Check SLURM
        if shutil.which("sinfo"):
            return self._discover_slurm()

        # Check Beaker
        if shutil.which("beaker"):
            return ClusterInfo(
                cluster_type="beaker",
                total_nodes=0,
                gpus_per_node=8,
                gpu_type="unknown",
                interconnect="unknown",
            )

        # Check Kubernetes
        if shutil.which("kubectl"):
            return ClusterInfo(
                cluster_type="kubernetes",
                total_nodes=0,
                gpus_per_node=8,
                gpu_type="unknown",
                interconnect="ethernet",
            )

        return None

    def _discover_slurm(self) -> Optional[ClusterInfo]:
        try:
            result = subprocess.run(
                ["sinfo", "-h", "-o", "%P|%D|%G"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                partitions = []
                total_nodes = 0
                gpu_type = "unknown"
                for line in result.stdout.strip().split("\n"):
                    parts = line.split("|")
                    if len(parts) >= 3:
                        partitions.append(parts[0].strip("*"))
                        total_nodes += int(parts[1])
                        if "gpu" in parts[2].lower():
                            gpu_type = parts[2]

                return ClusterInfo(
                    cluster_type="slurm",
                    total_nodes=total_nodes,
                    gpus_per_node=8,
                    gpu_type=gpu_type,
                    interconnect="infiniband",
                    available_partitions=partitions,
                )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None

    def _get_cuda_version(self) -> Optional[str]:
        try:
            result = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                import re
                match = re.search(r"release (\d+\.\d+)", result.stdout)
                if match:
                    return match.group(1)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        cuda_path = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_path and Path(cuda_path).exists():
            version_file = Path(cuda_path) / "version.txt"
            if version_file.exists():
                return version_file.read_text().strip().split()[-1]
        return None

    def _get_torch_version(self) -> Optional[str]:
        try:
            import torch
            return torch.__version__
        except ImportError:
            return None

    def _check_tools(self) -> Dict[str, bool]:
        tools = {}
        tools["nvidia_smi"] = shutil.which("nvidia-smi") is not None
        tools["torchrun"] = shutil.which("torchrun") is not None
        tools["git"] = shutil.which("git") is not None
        tools["wandb"] = shutil.which("wandb") is not None
        tools["beaker"] = shutil.which("beaker") is not None
        tools["slurm"] = shutil.which("sinfo") is not None
        tools["kubectl"] = shutil.which("kubectl") is not None

        try:
            import flash_attn  # noqa: F401
            tools["flash_attn"] = True
        except ImportError:
            tools["flash_attn"] = False

        try:
            import torch.distributed
            tools["nccl"] = torch.distributed.is_nccl_available()
        except (ImportError, AttributeError):
            tools["nccl"] = False

        return tools

    def _check_storage(self) -> Dict[str, Dict[str, Any]]:
        storage = {}
        paths_to_check = ["/", os.path.expanduser("~")]

        # Add common HPC paths
        for p in ["/scratch", "/data", "/shared", "/weka"]:
            if Path(p).exists():
                paths_to_check.append(p)

        for path in paths_to_check:
            try:
                usage = shutil.disk_usage(path)
                storage[path] = {
                    "total_gb": usage.total / (1024**3),
                    "free_gb": usage.free / (1024**3),
                    "used_pct": (usage.used / usage.total) * 100,
                }
            except (OSError, PermissionError):
                pass

        return storage

    def _check_network(self) -> Dict[str, str]:
        network = {}
        # Check for InfiniBand
        if Path("/sys/class/infiniband").exists():
            network["infiniband"] = "detected"
        else:
            network["infiniband"] = "not_found"

        # Check for NCCL socket interface
        nccl_socket = os.environ.get("NCCL_SOCKET_IFNAME", "")
        if nccl_socket:
            network["nccl_interface"] = nccl_socket

        return network

    def _identify_issues(self, report: EnvironmentReport) -> List[str]:
        issues = []
        if not report.gpus:
            issues.append("No GPUs detected — training will not work")
        if not report.cuda_version:
            issues.append("CUDA not found — install CUDA toolkit")
        if not report.torch_version:
            issues.append("PyTorch not installed")
        if not report.available_tools.get("flash_attn"):
            issues.append("flash-attn not installed (optional but recommended)")
        return issues

    def _generate_recommendations(self, report: EnvironmentReport) -> List[str]:
        recs = []
        if report.gpus and report.gpus[0].memory_total_gb < 40:
            recs.append("GPUs have <40GB memory — use FSDP with aggressive sharding")
        if report.cluster and report.cluster.interconnect != "infiniband":
            recs.append("No InfiniBand detected — multi-node training may be slow")
        if not report.available_tools.get("wandb"):
            recs.append("Install wandb for experiment tracking: pip install wandb")
        return recs
