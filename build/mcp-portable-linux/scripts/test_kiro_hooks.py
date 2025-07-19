import os
import json

HOOKS_DIR = os.path.join(os.path.dirname(__file__), '..', '.kiro', 'hooks')
REQUIRED_FIELDS = ["enabled", "name", "description", "version", "when", "then"]
REQUIRED_WHEN_FIELDS = ["type", "patterns"]
REQUIRED_THEN_FIELDS = ["type", "prompt"]


def validate_hook_file(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return False, f"JSON error: {e}"

    # Check top-level required fields
    for field in REQUIRED_FIELDS:
        if field not in data:
            return False, f"Missing required field: {field}"
    # Check 'when' fields
    for field in REQUIRED_WHEN_FIELDS:
        if field not in data["when"]:
            return False, f"Missing 'when' field: {field}"
    # Check 'then' fields
    for field in REQUIRED_THEN_FIELDS:
        if field not in data["then"]:
            return False, f"Missing 'then' field: {field}"
    # Check prompt is non-empty
    if not data["then"]["prompt"].strip():
        return False, "Prompt is empty"
    return True, "OK"


def main():
    print("Testing all .kiro/hooks/*.kiro.hook files...")
    for fname in os.listdir(HOOKS_DIR):
        if fname.endswith('.kiro.hook'):
            path = os.path.join(HOOKS_DIR, fname)
            valid, msg = validate_hook_file(path)
            status = "PASS" if valid else "FAIL"
            print(f"{fname}: {status} - {msg}")

if __name__ == "__main__":
    main() 