"""TENSORGRAPH CLI Tests — Command-Line Interface.

Comprehensive tests for the GCT-styled CLI:
- Command parsing
- Style utilities
- Output formatting
"""

from __future__ import annotations

import pytest


class TestStyleModule:
    """Test CLI styling utilities."""
    
    def test_color_codes_defined(self):
        """Color codes are defined."""
        from tensorgraph.cli.style import CYAN, AMBER, GREEN, RESET
        
        assert CYAN.startswith("\033")
        assert AMBER.startswith("\033")
        assert GREEN.startswith("\033")
        assert RESET == "\033[0m"
    
    def test_cyan_wrapper(self):
        """cyan() wraps text with color codes."""
        from tensorgraph.cli.style import cyan
        
        result = cyan("test")
        assert "test" in result
    
    def test_header_format(self):
        """header() produces industrial header block."""
        from tensorgraph.cli.style import header
        
        result = header("TEST HEADER", "OK")
        assert "TEST HEADER" in result
        assert ("━" in result or "═" in result)
    
    def test_section_format(self):
        """section() produces section divider."""
        from tensorgraph.cli.style import section
        
        result = section("SECTION TITLE")
        assert "SECTION TITLE" in result
        assert "─" in result
    
    def test_metric_format(self):
        """metric() produces labeled metric."""
        from tensorgraph.cli.style import metric, chrome
        
        result = metric("LABEL", "VALUE", chrome)
        assert "LABEL" in result
        assert "VALUE" in result
        assert "│" in result
    
    def test_metric_change_format(self):
        """metric_change() shows before/after."""
        from tensorgraph.cli.style import metric_change
        
        result = metric_change("BOXES", 10, 5)
        assert "10" in result
        assert "5" in result
        assert "→" in result
        assert "50%" in result
    
    def test_footer_format(self):
        """footer() produces GCT signature."""
        from tensorgraph.cli.style import footer
        
        result = footer()
        assert ("PACIFIC NORTHWEST" in result or "FRONTIER ENGINEERING" in result)
        assert ("━" in result or "═" in result)
    
    def test_banner_defined(self):
        """ASCII banner is defined."""
        from tensorgraph.cli.style import BANNER
        
        # Banner contains styled text
        assert len(BANNER) > 50  # Non-trivial content
    
    def test_trace_entry_format(self):
        """trace_entry() formats log entries."""
        from tensorgraph.cli.style import trace_entry
        
        result = trace_entry(5, "Rule applied")
        assert "[05]" in result
        assert "Rule applied" in result
    
    def test_success_format(self):
        """success() shows checkmark."""
        from tensorgraph.cli.style import success
        
        result = success("Done")
        assert "✓" in result
        assert "Done" in result
    
    def test_error_format(self):
        """error() shows X mark."""
        from tensorgraph.cli.style import error
        
        result = error("Failed")
        assert "✗" in result
        assert "Failed" in result


class TestMainCLI:
    """Test main CLI entry point."""
    
    def test_main_import(self):
        """Main module can be imported."""
        from tensorgraph.cli.main import main
        assert callable(main)
    
    def test_cmd_info_import(self):
        """Info command function exists."""
        from tensorgraph.cli.main import cmd_info
        assert callable(cmd_info)
    
    def test_cmd_optimize_import(self):
        """Optimize command function exists."""
        from tensorgraph.cli.main import cmd_optimize
        assert callable(cmd_optimize)
    
    def test_cmd_demo_import(self):
        """Demo command function exists."""
        from tensorgraph.cli.main import cmd_demo
        assert callable(cmd_demo)
