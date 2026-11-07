"""Tests that /new (and its /reset alias) clears session-scoped overrides."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

try:
    import pytest
    _has_pytest = True
except ImportError:
    _has_pytest = False
    class _DummyPytest:
        class mark:
            @staticmethod
            def asyncio(f):
                return f
    pytest = _DummyPytest()

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()

    session_key = build_session_key(_make_source())
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.reset_session.return_value = session_entry
    runner.session_store._entries = {session_key: session_entry}
    runner.session_store._generate_session_key.return_value = session_key
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache_lock = None  # disables _evict_cached_agent lock path
    runner._is_user_authorized = lambda _source: True
    runner._format_session_info = lambda: ""
    runner.session_store.set_model_override = MagicMock()

    return runner


@pytest.mark.asyncio
async def test_new_command_preserves_session_model_override():
    """/new keeps the explicitly selected model for the fresh conversation."""
    runner = _make_runner()
    session_key = build_session_key(_make_source())

    # Simulate a prior /model switch stored as a session override
    runner._session_model_overrides[session_key] = {
        "model": "gpt-4o",
        "provider": "openai",
        "api_key": "***",
        "base_url": "",
        "api_mode": "openai",
    }
    runner._session_reasoning_overrides[session_key] = {"enabled": True, "effort": "high"}
    runner._pending_model_notes[session_key] = "[Note: switched to gpt-4o.]"

    await runner._handle_reset_command(_make_event("/new"))

    assert runner._session_model_overrides[session_key]["model"] == "gpt-4o"
    runner.session_store.set_model_override.assert_called_once()
    assert session_key not in runner._session_reasoning_overrides
    assert session_key not in runner._pending_model_notes


@pytest.mark.asyncio
async def test_new_command_no_override_is_noop():
    """/new with no prior model override must not raise."""
    runner = _make_runner()
    session_key = build_session_key(_make_source())

    assert session_key not in runner._session_model_overrides
    assert session_key not in runner._session_reasoning_overrides

    await runner._handle_reset_command(_make_event("/new"))

    assert session_key not in runner._session_model_overrides
    assert session_key not in runner._session_reasoning_overrides


@pytest.mark.asyncio
async def test_new_command_preserves_own_session_model():
    """/new preserves this chat's model without touching another chat."""
    runner = _make_runner()
    session_key = build_session_key(_make_source())
    other_key = "other_session_key"

    runner._session_model_overrides[session_key] = {
        "model": "gpt-4o",
        "provider": "openai",
        "api_key": "sk-test",
        "base_url": "",
        "api_mode": "openai",
    }
    runner._session_model_overrides[other_key] = {
        "model": "claude-sonnet-4-6",
        "provider": "anthropic",
        "api_key": "***",
        "base_url": "",
        "api_mode": "anthropic",
    }
    runner._session_reasoning_overrides[session_key] = {"enabled": True, "effort": "high"}
    runner._session_reasoning_overrides[other_key] = {"enabled": True, "effort": "low"}
    runner._pending_model_notes[session_key] = "[Note: switched to gpt-4o.]"
    runner._pending_model_notes[other_key] = "[Note: switched to claude-sonnet-4-6.]"

    await runner._handle_reset_command(_make_event("/new"))

    assert runner._session_model_overrides[session_key]["model"] == "gpt-4o"
    assert other_key in runner._session_model_overrides
    assert session_key not in runner._session_reasoning_overrides
    assert other_key in runner._session_reasoning_overrides
    assert session_key not in runner._pending_model_notes
    assert other_key in runner._pending_model_notes


@pytest.mark.asyncio
async def test_new_command_preserves_from_old_entry_when_ram_cache_empty():
    """When in-memory overrides are empty (e.g. after restart), /new restores from old_entry.model_override."""
    runner = _make_runner()
    session_key = build_session_key(_make_source())

    # RAM cache is empty
    runner._session_model_overrides.clear()

    # But old_entry has a persisted model_override
    old_entry = runner.session_store._entries[session_key]
    old_entry.model_override = {
        "model": "glm-5.3-flash",
        "provider": "custom",
        "base_url": "https://example.com/v1",
    }

    await runner._handle_reset_command(_make_event("/new"))

    assert runner._session_model_overrides[session_key]["model"] == "glm-5.3-flash"
    runner.session_store.set_model_override.assert_called_once()


@pytest.mark.asyncio
async def test_new_command_banner_passes_inherited_override():
    """Verify that _reset_notice_session_info receives the inherited override."""
    runner = _make_runner()
    session_key = build_session_key(_make_source())

    runner._session_model_overrides[session_key] = {
        "model": "glm-5.3-flash",
        "provider": "custom",
    }

    captured_override = []

    def mock_notice_info(source, override=None):
        captured_override.append(override)
        return "mock_info"

    runner._reset_notice_session_info = mock_notice_info

    await runner._handle_reset_command(_make_event("/new"))

    assert len(captured_override) == 1
    assert captured_override[0]["model"] == "glm-5.3-flash"

