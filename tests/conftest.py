"""TENSORGRAPH Test Configuration — Chrome Metropolis Edition.

Pytest hooks for GCT-styled terminal output that comforts
uncles and aunties with respectful, high-caliber interfaces.
"""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.reports import TestReport
    from _pytest.terminal import TerminalReporter

# =============================================================================
# ANSI STYLING (Inline for portability)
# =============================================================================

CYAN = "\033[96m"
AMBER = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"
CHROME = "\033[97m"
STEEL = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"

WIDTH = 70


def _c(code: str) -> str:
    """Return color if stdout is a tty."""
    if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        return code
    return ""


# =============================================================================
# PYTEST HOOKS
# =============================================================================

def pytest_configure(config: "Config") -> None:
    """Configure pytest with GCT branding."""
    config.addinivalue_line("markers", "slow: marks tests as slow")


def pytest_report_header(config: "Config") -> list[str]:
    """Add GCT header to test report."""
    bar = "═" * WIDTH
    return [
        "",
        f"{_c(STEEL)}{bar}{_c(RESET)}",
        f"{_c(CYAN)}{_c(BOLD)}  TENSORGRAPH TEST SUITE                          [SYS:VERIFYING]{_c(RESET)}",
        f"{_c(STEEL)}{bar}{_c(RESET)}",
        f"{_c(STEEL)}  Diagrammatic Rewriting Compiler — Verification Protocol{_c(RESET)}",
        f"{_c(STEEL)}  Grand Challenge Technologies — Frontier Engineering{_c(RESET)}",
        "",
    ]


def pytest_runtest_logreport(report: "TestReport") -> None:
    """Custom logging for test results."""
    pass  # Let pytest handle default output, we enhance headers/footers


def pytest_terminal_summary(
    terminalreporter: "TerminalReporter", exitstatus: int, config: "Config"
) -> None:
    """Add GCT footer to test summary."""
    bar = "═" * WIDTH
    
    passed = len(terminalreporter.getreports("passed"))
    failed = len(terminalreporter.getreports("failed"))
    skipped = len(terminalreporter.getreports("skipped"))
    total = passed + failed + skipped
    
    if failed == 0:
        status = f"{_c(GREEN)}✓ ALL SYSTEMS OPERATIONAL{_c(RESET)}"
    else:
        status = f"{_c(RED)}✗ VERIFICATION FAILED{_c(RESET)}"
    
    terminalreporter.write_line("")
    terminalreporter.write_line(f"{_c(STEEL)}{'─' * WIDTH}{_c(RESET)}")
    terminalreporter.write_line(f"  {_c(STEEL)}SUMMARY{_c(RESET)}")
    terminalreporter.write_line(f"{_c(STEEL)}{'─' * WIDTH}{_c(RESET)}")
    terminalreporter.write_line(f"  {_c(STEEL)}PASSED{_c(RESET)}   │ {_c(GREEN)}{passed}{_c(RESET)}")
    terminalreporter.write_line(f"  {_c(STEEL)}FAILED{_c(RESET)}   │ {_c(RED)}{failed}{_c(RESET)}")
    terminalreporter.write_line(f"  {_c(STEEL)}SKIPPED{_c(RESET)}  │ {_c(AMBER)}{skipped}{_c(RESET)}")
    terminalreporter.write_line(f"  {_c(STEEL)}TOTAL{_c(RESET)}    │ {_c(CHROME)}{total}{_c(RESET)}")
    terminalreporter.write_line("")
    terminalreporter.write_line(f"  {status}")
    terminalreporter.write_line("")
    terminalreporter.write_line(f"{_c(STEEL)}{bar}{_c(RESET)}")
    terminalreporter.write_line(f"{_c(STEEL)}                    // FRONTIER ENGINEERING{_c(RESET)}")
    terminalreporter.write_line(f"{_c(STEEL)}{bar}{_c(RESET)}")


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def signature():
    """Provide a standard test signature."""
    from tensorgraph import Obj, Signature
    
    T = Obj("T")
    sig = Signature()
    sig.add("f", T, T)
    sig.add("g", T, T)
    sig.add("h", T, T)
    return sig


@pytest.fixture
def tensor_signature():
    """Provide a tensor-oriented signature."""
    from tensorgraph import Obj, Signature
    
    W = Obj("W")  # Weights
    X = Obj("X")  # Input
    Y = Obj("Y")  # Output
    
    sig = Signature()
    sig.add("Linear", W @ X, Y)
    sig.add("ReLU", Y, Y)
    sig.add("Dropout", Y, Y)
    sig.add("InjectLoRA", W, W)
    return sig


@pytest.fixture
def egraph(signature):
    """Provide an empty e-graph with standard signature."""
    from tensorgraph.egraph import EGraph
    return EGraph(signature)
