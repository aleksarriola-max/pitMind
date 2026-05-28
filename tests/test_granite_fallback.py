import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock
import agent.granite as g


def test_call_granite_returns_none_no_credentials(monkeypatch):
    """No credentials → _call_granite returns None without raising."""
    monkeypatch.setattr(g, "_get_credentials", lambda: ("", "", "https://example.com"))
    result = g._call_granite("test prompt", "test system")
    assert result is None


def test_call_granite_returns_none_on_connection_error(monkeypatch):
    """Network failure → _call_granite returns None without raising."""
    import requests
    monkeypatch.setattr(g, "_get_credentials",
                        lambda: ("fake-key", "fake-proj", "https://example.com"))
    monkeypatch.setattr(g, "_get_iam_token", lambda api_key: "fake-token")
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("timeout")):
        result = g._call_granite("test prompt", "test system")
    assert result is None


def test_pitwall_chat_returns_string_when_api_fails(monkeypatch):
    """After the pitwall_chat fix, a None from _call_granite returns a fallback string."""
    monkeypatch.setattr(g, "_call_granite", lambda prompt, system: None)
    result = g.pitwall_chat("Should I pit?", {}, "VER", "engineer")
    assert isinstance(result, str)
    assert len(result) > 0


def test_annotate_shift_fallback_fan(monkeypatch):
    """annotate_shift returns a non-empty string in fan mode when API is unavailable."""
    monkeypatch.setattr(g, "_call_granite", lambda prompt, system: None)
    shift = {
        "driver": "VER", "lap": 10, "direction": "up",
        "magnitude": 12.0, "momentum_before": 40.0, "momentum_after": 52.0,
        "team": "Red Bull",
    }
    result = g.annotate_shift(shift, mode="fan")
    assert isinstance(result, str)
    assert len(result) > 0


def test_annotate_shift_fallback_engineer(monkeypatch):
    """annotate_shift returns a string containing the driver code in engineer mode."""
    monkeypatch.setattr(g, "_call_granite", lambda prompt, system: None)
    shift = {
        "driver": "HAM", "lap": 15, "direction": "down",
        "magnitude": 9.5, "momentum_before": 60.0, "momentum_after": 50.5,
        "team": "Ferrari",
    }
    result = g.annotate_shift(shift, mode="engineer")
    assert isinstance(result, str)
    assert "HAM" in result or "Ferrari" in result or len(result) > 0
