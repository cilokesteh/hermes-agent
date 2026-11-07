#!/usr/bin/env python3
"""Regression test: the /new banner must report the SESSION override, not the config default.

Reproduces the user-visible bug: after `/model <x>` the banner showed
"◆ Model: gemini-3.8-flash / ◆ Provider: custom" (the config default)
while the session actually ran on the selected model.
"""
import os
import sys

sys.path.insert(0, "/home/ubuntu/.hermes/hermes-agent")
os.chdir("/home/ubuntu/.hermes/hermes-agent")

from gateway.run import GatewayRunner

runner = object.__new__(GatewayRunner)
runner.config = type("C", (), {"multiplex_profiles": False})()

override = {
    "model": "minimax-m3.1-flash",
    "provider": "custom:wildanoel",
    "base_url": "https://member.wildanoel.dev/v1",
    "api_key": "dummy",
}

banner = runner._format_session_info(override=override, source=None)
print(banner)

# 1) The banner must name the session model.
assert "minimax-m3.1-flash" in banner, f"session model missing from banner:\n{banner}"

# 2) The banner must name the session's PROVIDER, not the config's generic label.
assert "custom:wildanoel" in banner, f"session provider missing from banner:\n{banner}"
assert "\n◆ Provider: custom\n" not in banner, (
    f"banner clobbered the override provider with the config default:\n{banner}"
)

print("\nPASS: banner reports the session override (model + provider)")
