"""TENSORGRAPH Terminal Styling — Rustic Precision Aesthetic.

The Minderling's voice in the terminal.
Clean polished precision crafting amongst wild knowledge.

GCT Corporate Standard: Rustic North × Chrome Metropolis
"""

from __future__ import annotations

import sys

# =============================================================================
# RUSTIC PRECISION COLOR SYSTEM
# =============================================================================

# Deep Forest — The grounding darkness
DEEP_FOREST = "\033[38;2;13;18;16m"   # #0d1210

# Cedar Core — Warm accents, the crafter's touch
CEDAR = "\033[38;2;196;149;106m"       # #c4956a
CEDAR_BRIGHT = "\033[38;2;220;175;130m"

# Lichen Glow — The pulse of awareness
LICHEN = "\033[38;2;127;204;176m"      # #7fccb0
LICHEN_BRIGHT = "\033[38;2;150;230;200m"

# Chrome — Precision readouts
CHROME = "\033[38;2;212;216;220m"      # #d4d8dc
CHROME_DIM = "\033[38;2;138;145;152m"  # #8a9198

# Status Colors
SUCCESS = "\033[38;2;127;204;176m"     # Lichen
WARNING = "\033[38;2;196;149;106m"     # Cedar
ERROR = "\033[38;2;180;80;80m"         # Muted red
TRACE = "\033[38;2;138;145;152m"       # Chrome dim

# Formatting
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# Legacy color aliases
CYAN = LICHEN
AMBER = CEDAR
GREEN = SUCCESS

# =============================================================================
# DETECTION
# =============================================================================

def supports_color() -> bool:
    """Check if terminal supports ANSI colors."""
    if not hasattr(sys.stdout, "isatty"):
        return False
    if not sys.stdout.isatty():
        return False
    return True

_COLOR_ENABLED = supports_color()

def _c(code: str) -> str:
    """Return color code if supported, empty string otherwise."""
    return code if _COLOR_ENABLED else ""

# =============================================================================
# PRIMITIVES — The Minderling's Palette
# =============================================================================

def lichen(text: str) -> str:
    """Primary accent — awareness, success, active states."""
    return f"{_c(LICHEN)}{text}{_c(RESET)}"

def cedar(text: str) -> str:
    """Warm accent — highlights, warnings, the crafter's hand."""
    return f"{_c(CEDAR)}{text}{_c(RESET)}"

def chrome(text: str) -> str:
    """Neutral precision — body text, metrics."""
    return f"{_c(CHROME)}{text}{_c(RESET)}"

def dim(text: str) -> str:
    """Subdued text — metadata, timestamps."""
    return f"{_c(CHROME_DIM)}{text}{_c(RESET)}"

def green(text: str) -> str:
    """Success indicator."""
    return f"{_c(SUCCESS)}{text}{_c(RESET)}"

def red(text: str) -> str:
    """Error indicator."""
    return f"{_c(ERROR)}{text}{_c(RESET)}"

def bold(text: str) -> str:
    return f"{_c(BOLD)}{text}{_c(RESET)}"

# Legacy aliases for compatibility
cyan = lichen
amber = cedar
steel = dim

# =============================================================================
# COMPONENTS — Rustic Precision
# =============================================================================

WIDTH = 68

def header(title: str, status: str = "READY") -> str:
    """Forest-inspired header block."""
    bar = "━" * WIDTH
    status_str = f"[{status}]"
    padding = WIDTH - len(title) - len(status_str) - 4
    content = f"  {title}{' ' * padding}{status_str}"
    return f"""
{_c(CHROME_DIM)}{bar}{_c(RESET)}
{_c(LICHEN)}{_c(BOLD)}{content}{_c(RESET)}
{_c(CHROME_DIM)}{bar}{_c(RESET)}
"""

def section(title: str) -> str:
    """Section divider with organic flow."""
    bar = "─" * WIDTH
    return f"""
{_c(CHROME_DIM)}{bar}{_c(RESET)}
  {_c(CEDAR)}{title}{_c(RESET)}
{_c(CHROME_DIM)}{bar}{_c(RESET)}"""

def divider() -> str:
    """Simple divider line."""
    return f"{_c(CHROME_DIM)}{'─' * WIDTH}{_c(RESET)}"

def footer() -> str:
    """GCT signature footer — Rustic North."""
    bar = "━" * WIDTH
    sig = "CRAFTED IN THE PACIFIC NORTHWEST"
    padding = (WIDTH - len(sig)) // 2
    return f"""
{_c(CHROME_DIM)}{bar}{_c(RESET)}
{_c(CHROME_DIM)}{' ' * padding}{sig}{_c(RESET)}
{_c(CHROME_DIM)}{bar}{_c(RESET)}
"""

def metric(label: str, value: str, color_fn=chrome) -> str:
    """Labeled metric display."""
    return f"  {_c(CHROME_DIM)}{label:10}{_c(RESET)} │ {color_fn(value)}"

def metric_change(label: str, before: int, after: int) -> str:
    """Metric with before/after and percentage."""
    if before > 0:
        pct = ((before - after) / before) * 100
        arrow = "▼" if after < before else "▲" if after > before else "─"
        color = green if after < before else red if after > before else chrome
        change = f"({arrow} {abs(pct):.0f}%)"
    else:
        change = ""
        color = chrome
    return f"  {_c(CHROME_DIM)}{label:10}{_c(RESET)} │ {before} → {color(str(after))}  {_c(CHROME_DIM)}{change}{_c(RESET)}"

def trace_entry(index: int, message: str) -> str:
    """Trace log entry."""
    return f"  {_c(CHROME_DIM)}[{index:02d}]{_c(RESET)} {message}"

def success(message: str) -> str:
    """Success indicator — the Minderling approves."""
    return f"  {_c(LICHEN)}✓{_c(RESET)} {message}"

def error(message: str) -> str:
    """Error indicator."""
    return f"  {_c(ERROR)}✗{_c(RESET)} {message}"

def status_dot(status: str = "OPERATIONAL") -> str:
    """Status indicator with organic glow."""
    return f"{_c(LICHEN)}●{_c(RESET)} {status}"

# =============================================================================
# BANNER — The Minderling's Welcome
# =============================================================================

def get_banner() -> str:
    """Generate the TENSORGRAPH ASCII banner in Rustic Precision style."""
    l = _c(LICHEN)        # Lichen glow for logo
    b = _c(BOLD)
    r = _c(RESET)
    w = _c(CHROME)        # Chrome for tagline
    d = _c(CHROME_DIM)    # Dim for secondary
    c = _c(CEDAR)         # Cedar for accent
    
    return f"""
{l}{b} █████╗ ███████╗████████╗██╗  ██╗███████╗██████╗ ██████╗ ██████╗ {r}
{l}{b}██╔══██╗██╔════╝╚══██╔══╝██║  ██║██╔════╝██╔══██╗╚════██╗██╔══██╗{r}
{l}{b}███████║█████╗     ██║   ███████║█████╗  ██████╔╝ █████╔╝██║  ██║{r}
{l}{b}██╔══██║██╔══╝     ██║   ██╔══██║██╔══╝  ██╔══██╗██╔═══╝ ██║  ██║{r}
{l}{b}██║  ██║███████╗   ██║   ██║  ██║███████╗██║  ██║███████╗██████╔╝{r}
{l}{b}╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝╚═════╝ {r}
{w}Diagrammatic Rewriting Compiler{r}
{d}Grand Challenge Technologies{r} — {c}Knowledge from beyond the forest{r}
"""

# Keep BANNER for backward compatibility
BANNER = get_banner()

def print_banner():
    """Print the TENSORGRAPH ASCII banner."""
    print(get_banner())

# =============================================================================
# MINDER AWARENESS — CLI version of the insight callout
# =============================================================================

def minder_says(message: str) -> str:
    """The Minderling speaks — a moment of wisdom in the terminal."""
    bar = "─" * WIDTH
    return f"""
{_c(CHROME_DIM)}{bar}{_c(RESET)}
  {_c(CEDAR)}THE MINDERLING SPEAKS{_c(RESET)}
  {_c(CEDAR)}{message}{_c(RESET)}
{_c(CHROME_DIM)}{bar}{_c(RESET)}
"""
