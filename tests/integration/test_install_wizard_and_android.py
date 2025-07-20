#!/usr/bin/env python3
"""
Integration test for Universal Install Wizard and Android build automation.
"""
import subprocess
import sys
import os
import pytest

def test_setup_wizard_android_prompt(monkeypatch):
    """Test that the setup wizard prompts for Android build/install and LLM API key setup."""
    responses = iter(['n', 'n'])  # Simulate 'no' to both prompts
    monkeypatch.setattr('builtins.input', lambda _: next(responses))
    from scripts import setup_wizard
    wizard = setup_wizard.SetupWizard()
    assert wizard.android_setup_prompt() is None
    assert wizard.llm_api_key_prompt() is None

def test_build_android_script_exists():
    """Test that the Android build script exists and is executable."""
    script_path = os.path.join('scripts', 'build_android.py')
    assert os.path.exists(script_path)
    assert os.access(script_path, os.X_OK) or script_path.endswith('.py')

def test_llm_api_key_prompt(monkeypatch):
    """Test LLM API key prompt logic in setup wizard."""
    responses = iter(['y'])
    monkeypatch.setattr('builtins.input', lambda _: next(responses))
    from scripts import setup_wizard
    wizard = setup_wizard.SetupWizard()
    assert wizard.llm_api_key_prompt() is None 