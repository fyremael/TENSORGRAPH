"""Tests for Colab CLI Connection Monitor."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch
import pytest

from tensorgraph.cli.colab_monitor import (
    ColabConnectionConfig,
    ColabMetrics,
    ColabMonitor,
)
from tensorgraph.cli.main import cmd_colab


class TestColabMonitorParsers:
    """Test metrics parsing utilities."""

    def test_parse_nvidia_smi_valid(self):
        """Parse valid nvidia-smi csv output."""
        output = "Tesla T4, 32, 4096, 15360\n"
        stats = ColabMonitor.parse_nvidia_smi(output)

        assert stats["gpu_name"] == "Tesla T4"
        assert stats["gpu_utilization"] == 32.0
        assert stats["gpu_memory_used_mb"] == 4096.0
        assert stats["gpu_memory_total_mb"] == 15360.0

    def test_parse_nvidia_smi_empty(self):
        """Parse empty or invalid nvidia-smi output gracefully."""
        stats = ColabMonitor.parse_nvidia_smi("")
        assert stats["gpu_name"] == "N/A"
        assert stats["gpu_utilization"] == 0.0

    def test_parse_system_stats_valid(self):
        """Parse system stats string."""
        raw = "CPU: 18.5% | RAM: 4.5/12.7 GB | DISK: 30.2/100.0 GB | UPTIME: 7200"
        stats = ColabMonitor.parse_system_stats(raw)

        assert stats["cpu_utilization"] == 18.5
        assert stats["ram_used_gb"] == 4.5
        assert stats["ram_total_gb"] == 12.7
        assert stats["disk_used_gb"] == 30.2
        assert stats["disk_total_gb"] == 100.0
        assert stats["uptime_seconds"] == 7200.0

    def test_parse_system_stats_empty(self):
        """Parse empty system stats string."""
        stats = ColabMonitor.parse_system_stats("")
        assert stats["cpu_utilization"] == 0.0
        assert stats["ram_used_gb"] == 0.0


class TestColabMonitorProbes:
    """Test socket probing and metrics collection."""

    @patch("socket.create_connection")
    def test_probe_tcp_connection_success(self, mock_create):
        """Probe TCP connection returns success and latency."""
        mock_sock = MagicMock()
        mock_create.return_value = mock_sock

        connected, latency, err = ColabMonitor.probe_tcp_connection("127.0.0.1", 22, timeout=1.0)
        assert connected is True
        assert latency >= 0.0
        assert err == ""
        mock_sock.close.assert_called_once()

    @patch("socket.create_connection")
    def test_probe_tcp_connection_failure(self, mock_create):
        """Probe TCP connection handles socket error."""
        mock_create.side_effect = TimeoutError("Connection timed out")

        connected, latency, err = ColabMonitor.probe_tcp_connection("127.0.0.1", 9999, timeout=1.0)
        assert connected is False
        assert "timed out" in err.lower()

    @patch.object(ColabMonitor, "probe_tcp_connection")
    def test_collect_metrics_disconnected(self, mock_probe):
        """collect_metrics handles disconnected state."""
        mock_probe.return_value = (False, 500.0, "Connection refused")
        monitor = ColabMonitor()
        config = ColabConnectionConfig(host="invalid.host", port=22)

        metrics = monitor.collect_metrics(config)
        assert metrics.status == "DISCONNECTED"
        assert metrics.error_message == "Connection refused"

    @patch.object(ColabMonitor, "_query_remote_telemetry")
    @patch.object(ColabMonitor, "probe_tcp_connection")
    def test_collect_metrics_connected(self, mock_probe, mock_telemetry):
        """collect_metrics populates telemetry when connected."""
        mock_probe.return_value = (True, 45.0, "")
        mock_telemetry.return_value = {
            "nvidia_smi": "NVIDIA A100-SXM4-40GB, 15, 2048, 40960",
            "sys_stats": "CPU: 10.0% | RAM: 8.0/32.0 GB | DISK: 50.0/200.0 GB | UPTIME: 3600"
        }
        monitor = ColabMonitor()
        config = ColabConnectionConfig(host="127.0.0.1", port=22)

        metrics = monitor.collect_metrics(config, heartbeat_count=3)
        assert metrics.status == "CONNECTED"
        assert metrics.gpu_name == "NVIDIA A100-SXM4-40GB"
        assert metrics.gpu_utilization == 15.0
        assert metrics.cpu_utilization == 10.0
        assert metrics.ram_used_gb == 8.0
        assert metrics.heartbeat_count == 3

    def test_render_status_report(self):
        """render_status_report produces non-empty styled output."""
        monitor = ColabMonitor()
        config = ColabConnectionConfig(host="colab.test", port=22)
        metrics = ColabMetrics(
            status="CONNECTED",
            latency_ms=25.4,
            gpu_name="Tesla T4",
            gpu_utilization=40.0,
            gpu_memory_used_mb=3000.0,
            gpu_memory_total_mb=15000.0,
            cpu_utilization=15.0,
            ram_used_gb=6.0,
            ram_total_gb=13.0,
            disk_used_gb=20.0,
            disk_total_gb=80.0,
            heartbeat_count=5,
            last_heartbeat="12:00:00",
        )

        report = monitor.render_status_report(metrics, config)
        assert "COLAB CONNECTION MONITOR" in report
        assert "CONNECTED" in report
        assert "Tesla T4" in report
        assert "25.40 ms" in report


class TestColabCLICommand:
    """Test CLI subcommand integration."""

    @patch("tensorgraph.cli.colab_monitor.ColabMonitor.collect_metrics")
    def test_cmd_colab_status_connected(self, mock_collect, capsys):
        """cmd_colab status returns 0 when connected."""
        mock_collect.return_value = ColabMetrics(status="CONNECTED", latency_ms=10.0)

        args = argparse.Namespace(
            command="colab",
            action="status",
            host="localhost",
            port=22,
            user="root",
            password="antigravity",
            timeout=1.0,
            interval=10.0,
            no_keep_alive=False,
        )

        code = cmd_colab(args)
        assert code == 0
        captured = capsys.readouterr().out
        assert "COLAB CONNECTION MONITOR" in captured

    @patch("tensorgraph.cli.colab_monitor.ColabMonitor.collect_metrics")
    def test_cmd_colab_status_disconnected(self, mock_collect, capsys):
        """cmd_colab status returns 1 when disconnected."""
        mock_collect.return_value = ColabMetrics(status="DISCONNECTED", error_message="Refused")

        args = argparse.Namespace(
            command="colab",
            action="status",
            host="localhost",
            port=22,
            user="root",
            password="antigravity",
            timeout=1.0,
            interval=10.0,
            no_keep_alive=False,
        )

        code = cmd_colab(args)
        assert code == 1
        captured = capsys.readouterr().out
        assert "DISCONNECTED" in captured

    @patch("tensorgraph.cli.colab_monitor.ColabMonitor.collect_metrics")
    def test_cmd_colab_monitor_once(self, mock_collect, capsys):
        """cmd_colab monitor with once=True runs single iteration."""
        mock_collect.return_value = ColabMetrics(status="CONNECTED", latency_ms=12.0)

        args = argparse.Namespace(
            command="colab",
            action="monitor",
            host="localhost",
            port=22,
            user="root",
            password="antigravity",
            timeout=1.0,
            interval=1.0,
            no_keep_alive=False,
            once=True,
        )

        code = cmd_colab(args)
        assert code == 0
        captured = capsys.readouterr().out
        assert "COLAB CONNECTION MONITORING" in captured
