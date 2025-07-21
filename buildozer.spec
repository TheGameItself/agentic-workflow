[app]
title = MCP Agentic Workflow
package.name = mcp_agentic_workflow
package.domain = org.agenticworkflow
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json,md
version = 1.0.0
requirements = python3,kivy,requests,numpy,sqlalchemy
orientation = portrait
fullscreen = 1
android.permissions = INTERNET,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE
android.api = 33
android.minapi = 21
android.ndk = 23b
android.arch = armeabi-v7a,arm64-v8a,x86,x86_64
android.allow_backup = True
android.debug = 1
android.logcat_filters = *:S python:D
android.entrypoint = main.py
android.icon = %(source.dir)s/icon.png
android.presplash = %(source.dir)s/presplash.png
# LLM API/network compatibility
# Add any additional requirements for LLM API support here
# e.g., openai, anthropic, cohere, etc.

[buildozer]
log_level = 2
warn_on_root = 1

# (Optional) Keystore for release builds
# android.release_keystore = /path/to/keystore
# android.release_keyalias = mykeyalias
# android.release_keyalias_passwd = mypassword 