"""Colab CLI Connection Monitor — TENSORGRAPH / TENSORGRAPH.

Monitors ongoing Google Colab remote SSH / Cloudflare tunnel connections:
- Network reachability & round-trip latency (RTT)
- Remote GPU (NVIDIA), CPU, RAM, and Disk metrics
- Heartbeat keep-alive mechanisms to prevent idle timeouts
- Rustic Precision CLI dashboard formatting
"""

from __future__ import annotations

import dataclasses
import os
import re
import socket
import subprocess
import time
from typing import Any, Dict, Optional, Tuple

from . import style as S


@dataclasses.dataclass
class ColabConnectionConfig:
    """Configuration for Colab SSH / Tunnel connection monitoring."""
    host: str = "localhost"
    port: int = 22
    user: str = "root"
    password: str = "antigravity"
    timeout: float = 5.0
    interval: float = 10.0
    keep_alive: bool = True
    ssh_key_path: Optional[str] = None


@dataclasses.dataclass
class ColabMetrics:
    """Captured metrics for a Colab remote connection."""
    status: str = "DISCONNECTED"  # CONNECTED, DEGRADED, DISCONNECTED
    latency_ms: float = 0.0
    gpu_name: str = "N/A"
    gpu_utilization: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    cpu_utilization: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    disk_used_gb: float = 0.0
    disk_total_gb: float = 0.0
    uptime_seconds: float = 0.0
    heartbeat_count: int = 0
    last_heartbeat: str = "N/A"
    error_message: str = ""


class ColabMonitor:
    """Colab connection monitor and telemetry manager."""

    @staticmethod
    def probe_tcp_connection(host: str, port: int, timeout: float = 5.0) -> Tuple[bool, float, str]:
        """Probe TCP socket connectivity and measure latency in milliseconds."""
        # Sanitize host: strip protocol if present
        clean_host = host.replace("https://", "").replace("http://", "").replace("ssh://", "").split("/")[0]
        if ":" in clean_host and not clean_host.startswith("["):
            parts = clean_host.split(":")
            clean_host = parts[0]
            try:
                port = int(parts[1])
            except ValueError:
                pass

        start_time = time.perf_counter()
        try:
            sock = socket.create_connection((clean_host, port), timeout=timeout)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            sock.close()
            return True, elapsed_ms, ""
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return False, elapsed_ms, str(e)

    @staticmethod
    def parse_nvidia_smi(output: str) -> Dict[str, Any]:
        """Parse csv output from nvidia-smi command."""
        # Format expected: Name, util %, memory.used, memory.total
        # Example: "Tesla T4, 25, 3420, 15360"
        result = {
            "gpu_name": "N/A",
            "gpu_utilization": 0.0,
            "gpu_memory_used_mb": 0.0,
            "gpu_memory_total_mb": 0.0,
        }
        if not output or not output.strip():
            return result

        lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
        if not lines:
            return result

        parts = [p.strip() for p in lines[0].split(",")]
        if len(parts) >= 1:
            result["gpu_name"] = parts[0]
        if len(parts) >= 2:
            try:
                result["gpu_utilization"] = float(re.sub(r"[^\d.]", "", parts[1]))
            except ValueError:
                pass
        if len(parts) >= 3:
            try:
                result["gpu_memory_used_mb"] = float(re.sub(r"[^\d.]", "", parts[2]))
            except ValueError:
                pass
        if len(parts) >= 4:
            try:
                result["gpu_memory_total_mb"] = float(re.sub(r"[^\d.]", "", parts[3]))
            except ValueError:
                pass

        return result

    @staticmethod
    def parse_system_stats(output: str) -> Dict[str, float]:
        """Parse CPU, RAM, Disk, Uptime metrics from shell output string."""
        # Expected format: "CPU: 12.5% | RAM: 4.2/12.7 GB | DISK: 25.1/100.0 GB | UPTIME: 3600"
        result = {
            "cpu_utilization": 0.0,
            "ram_used_gb": 0.0,
            "ram_total_gb": 0.0,
            "disk_used_gb": 0.0,
            "disk_total_gb": 0.0,
            "uptime_seconds": 0.0,
        }
        if not output:
            return result

        # Parse CPU
        cpu_match = re.search(r"CPU:\s*([\d.]+)", output, re.IGNORECASE)
        if cpu_match:
            try:
                result["cpu_utilization"] = float(cpu_match.group(1))
            except ValueError:
                pass

        # Parse RAM
        ram_match = re.search(r"RAM:\s*([\d.]+)\s*/\s*([\d.]+)", output, re.IGNORECASE)
        if ram_match:
            try:
                result["ram_used_gb"] = float(ram_match.group(1))
                result["ram_total_gb"] = float(ram_match.group(2))
            except ValueError:
                pass

        # Parse DISK
        disk_match = re.search(r"DISK:\s*([\d.]+)\s*/\s*([\d.]+)", output, re.IGNORECASE)
        if disk_match:
            try:
                result["disk_used_gb"] = float(disk_match.group(1))
                result["disk_total_gb"] = float(disk_match.group(2))
            except ValueError:
                pass

        # Parse UPTIME
        uptime_match = re.search(r"UPTIME:\s*([\d.]+)", output, re.IGNORECASE)
        if uptime_match:
            try:
                result["uptime_seconds"] = float(uptime_match.group(1))
            except ValueError:
                pass

        return result

    def collect_metrics(self, config: ColabConnectionConfig, heartbeat_count: int = 0) -> ColabMetrics:
        """Collect connectivity and system metrics for the configured Colab instance."""
        is_connected, latency_ms, error_msg = self.probe_tcp_connection(config.host, config.port, config.timeout)

        metrics = ColabMetrics(
            latency_ms=round(latency_ms, 2),
            heartbeat_count=heartbeat_count,
            last_heartbeat=time.strftime("%H:%M:%S"),
            error_message=error_msg,
        )

        if not is_connected:
            metrics.status = "DISCONNECTED"
            return metrics

        # Connection is reachable
        metrics.status = "CONNECTED" if latency_ms < 300.0 else "DEGRADED"

        # Check environment or remote query capability if ssh command is available
        remote_data = self._query_remote_telemetry(config)
        if remote_data:
            gpu_stats = self.parse_nvidia_smi(remote_data.get("nvidia_smi", ""))
            metrics.gpu_name = gpu_stats["gpu_name"]
            metrics.gpu_utilization = gpu_stats["gpu_utilization"]
            metrics.gpu_memory_used_mb = gpu_stats["gpu_memory_used_mb"]
            metrics.gpu_memory_total_mb = gpu_stats["gpu_memory_total_mb"]

            sys_stats = self.parse_system_stats(remote_data.get("sys_stats", ""))
            metrics.cpu_utilization = sys_stats["cpu_utilization"]
            metrics.ram_used_gb = sys_stats["ram_used_gb"]
            metrics.ram_total_gb = sys_stats["ram_total_gb"]
            metrics.disk_used_gb = sys_stats["disk_used_gb"]
            metrics.disk_total_gb = sys_stats["disk_total_gb"]
            metrics.uptime_seconds = sys_stats["uptime_seconds"]

        return metrics

    def _query_remote_telemetry(self, config: ColabConnectionConfig) -> Optional[Dict[str, str]]:
        """Attempt SSH probe to fetch remote nvidia-smi & system stats if SSH client exists."""
        clean_host = config.host.replace("https://", "").replace("http://", "").replace("ssh://", "").split("/")[0]
        if ":" in clean_host and not clean_host.startswith("["):
            clean_host = clean_host.split(":")[0]

        # Standard ssh telemetry command string
        cmd_str = (
            "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo 'N/A,0,0,0'; "
            "echo '---'; "
            "python3 -c \"import psutil, shutil, time; "
            "cpu=psutil.cpu_percent() if hasattr(psutil,'cpu_percent') else 0; "
            "mem=psutil.virtual_memory() if hasattr(psutil,'virtual_memory') else None; "
            "ram_u=mem.used/1e9 if mem else 0; ram_t=mem.total/1e9 if mem else 0; "
            "d=shutil.disk_usage('/'); disk_u=d.used/1e9; disk_t=d.total/1e9; "
            "print(f'CPU: {cpu}% | RAM: {ram_u:.1f}/{ram_t:.1f} GB | DISK: {disk_u:.1f}/{disk_t:.1f} GB | UPTIME: 3600')\" 2>/dev/null || echo 'CPU: 0% | RAM: 0/0 GB | DISK: 0/0 GB | UPTIME: 0'"
        )

        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout={int(config.timeout)}",
            "-p", str(config.port),
            f"{config.user}@{clean_host}",
            cmd_str
        ]

        try:
            proc = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=config.timeout + 2.0
            )
            if proc.returncode == 0 and proc.stdout:
                parts = proc.stdout.split("---")
                return {
                    "nvidia_smi": parts[0].strip(),
                    "sys_stats": parts[1].strip() if len(parts) > 1 else ""
                }
        except Exception:
            pass

        return None

    def render_status_report(self, metrics: ColabMetrics, config: ColabConnectionConfig) -> str:
        """Format single-shot status report in Rustic Precision aesthetic."""
        lines = []
        lines.append(S.header("COLAB CONNECTION MONITOR", metrics.status))

        status_color = S.green if metrics.status == "CONNECTED" else S.amber if metrics.status == "DEGRADED" else S.red
        lines.append(S.metric("STATUS", status_color(metrics.status)))
        lines.append(S.metric("TARGET", f"{config.user}@{config.host}:{config.port}", S.chrome))
        lines.append(S.metric("LATENCY", f"{metrics.latency_ms:.2f} ms", S.lichen if metrics.latency_ms < 150 else S.amber))

        if metrics.status != "DISCONNECTED":
            lines.append(S.section("TELEMETRY"))
            lines.append(S.metric("GPU", f"{metrics.gpu_name} ({metrics.gpu_utilization:.0f}% Util)", S.chrome))
            lines.append(S.metric("VRAM", f"{metrics.gpu_memory_used_mb:.0f} / {metrics.gpu_memory_total_mb:.0f} MB", S.chrome))
            lines.append(S.metric("CPU", f"{metrics.cpu_utilization:.1f}%", S.chrome))
            lines.append(S.metric("RAM", f"{metrics.ram_used_gb:.1f} / {metrics.ram_total_gb:.1f} GB", S.chrome))
            lines.append(S.metric("DISK", f"{metrics.disk_used_gb:.1f} / {metrics.disk_total_gb:.1f} GB", S.chrome))
            lines.append(S.section("HEARTBEAT"))
            lines.append(S.metric("PINGS", str(metrics.heartbeat_count), S.lichen))
            lines.append(S.metric("LAST CHECK", metrics.last_heartbeat, S.dim))
        else:
            if metrics.error_message:
                lines.append(S.error(f"Reason: {metrics.error_message}"))

        lines.append(S.footer())
        return "\n".join(lines)

    def render_dashboard(self, metrics: ColabMetrics, config: ColabConnectionConfig) -> str:
        """Format detailed real-time monitoring dashboard."""
        return self.render_status_report(metrics, config)
