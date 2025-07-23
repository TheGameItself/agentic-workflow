#!/usr/bin/env python3
'''
# MMCP-START
[ ] #".root"# {protocol:"MCP", version:"1.2.1", standard:"PFSUS+EARS+LambdaJSON"}
# MMCP Proof-Counterproof Script
## {type:Meta, author:"Kalxi", license:"MIT", last_modified:"2025-07-21T00:00:00Z", id:"MMCP.Tools.MMCP-TOOL-001"}

## {type:Tool, id:"MMCP-PROOF", desc:"Automated proof-counterproof script for validating MMCP documents using megalithic regex patterns."}
# MMCP-END
'''

import re
import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

def load_megalithic_regex() -> Tuple[str, str]:
    """Load megalithic regex patterns from config or use defaults."""
    config_path = Path(__file__).parent.parent / ".legend.mmcp" / "config.mmcp.json"
    
    # Default patterns
    valid_pattern = r"^(\[ \] #\"\.root\"#.*|#+\s.*|---.*|##\s*\{.*\}.*|@\{.*\}.*|%%.*|<mmcp:nested.*>.*</mmcp:nested>|\s*[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*.*|\s*\"[a-zA-Z_][a-zA-Z0-9_]*\"\s*:\s*.*|\s*\[.*\]\s*|\s*\{.*\}\s*|\s*[0-9]+\s*|\s*\".*\"\s*|\s*'.*'\s*|\s*true\s*|\s*false\s*|\s*null\s*|\s*undefined\s*|\s*NaN\s*|\s*Infinity\s*|\s*-Infinity\s*|\s*λ:.*|\s*$)"
    invalid_pattern = r"^(?!(\[ \] #\"\.root\"#.*|#+\s.*|---.*|##\s*\{.*\}.*|@\{.*\}.*|%%.*|<mmcp:nested.*>.*</mmcp:nested>|\s*[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*.*|\s*\"[a-zA-Z_][a-zA-Z0-9_]*\"\s*:\s*.*|\s*\[.*\]\s*|\s*\{.*\}\s*|\s*[0-9]+\s*|\s*\".*\"\s*|\s*'.*'\s*|\s*true\s*|\s*false\s*|\s*null\s*|\s*undefined\s*|\s*NaN\s*|\s*Infinity\s*|\s*-Infinity\s*|\s*λ:.*|\s*$)).*"
    
    # Try to load from config
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding='utf-8'))
            if "megalithicRegex" in config:
                valid_pattern = config["megalithicRegex"].get("valid", valid_pattern)
                invalid_pattern = config["megalithicRegex"].get("invalid", invalid_pattern)
        except Exception as e:
            print(f"Warning: Failed to load megalithic regex from config: {e}", file=sys.stderr)
    
    return valid_pattern, invalid_pattern

def validate_with_megalithic_regex(content: str) -> Tuple[bool, List[str]]:
    """Validate content using megalithic regex patterns."""
    valid_pattern, invalid_pattern = load_megalithic_regex()
    
    invalid_lines = []
    valid = True
    
    for i, line in enumerate(content.splitlines(), 1):
        if line.strip() and not re.match(valid_pattern, line):
            if re.match(invalid_pattern, line):
                invalid_lines.append(f"Line {i}: {line}")
                valid = False
    
    return valid, invalid_lines

def generate_proof_report(file_path: Path, valid: bool, invalid_lines: List[str]) -> Dict[str, Any]:
    """Generate a proof report for the validated file."""
    line_count = len(file_path.read_text(encoding='utf-8').splitlines())
    report = {
        "file": str(file_path),
        "valid": valid,
        "timestamp": datetime.now().isoformat(),
        "invalid_lines": invalid_lines,
        "line_count": line_count,
        "valid_percentage": 100 if valid else (1 - len(invalid_lines) / line_count) * 100
    }
    return report

def extract_mmcp_from_wrapped(file_path: Path) -> Tuple[bool, str]:
    """Extract MMCP content from a wrapped file based on extension."""
    content = file_path.read_text(encoding='utf-8')
    file_ext = file_path.suffix.lower()
    
    # Check if it's already an unwrapped MMCP file
    if file_ext == '.mmd' and '[ ] #".root"#' in content:
        return False, content
    
    # Handle Python files
    if file_ext == '.py':
        match = re.search(r"'''\s*# MMCP-START\s*([\s\S]*?)\s*# MMCP-END\s*'''", content, re.DOTALL)
        if match:
            return True, match.group(1).strip()
    
    # Handle Markdown files
    elif file_ext == '.md':
        match = re.search(r"```mmcp\s*<!-- MMCP-START -->\s*([\s\S]*?)\s*<!-- MMCP-END -->\s*```", content, re.DOTALL)
        if match:
            return True, match.group(1).strip()
    
    # Handle JavaScript/SQL files
    elif file_ext in ['.js', '.sql']:
        match = re.search(r"/\*\s*\* MMCP-START\s*([\s\S]*?)\s*\* MMCP-END\s*\*/", content, re.DOTALL)
        if match:
            return True, match.group(1).strip()
    
    # Handle JSON files
    elif file_ext == '.json':
        try:
            data = json.loads(content)
            if "__mmcp" in data and "content" in data["__mmcp"]:
                return True, data["__mmcp"]["content"]
        except json.JSONDecodeError:
            pass
    
    # Handle TOML files
    elif file_ext == '.toml':
        match = re.search(r"# MMCP-START\s*\[mmcp\]\s*content = '''\s*([\s\S]*?)\s*'''\s*# MMCP-END", content, re.DOTALL)
        if match:
            return True, match.group(1).strip()
    
    # If no specific format matched or extraction failed, return original content
    return False, content

def main():
    if len(sys.argv) < 2:
        print("Usage: mmcp_proof_counterproof.py <file_path> [output_report_path]")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist.")
        sys.exit(1)
    
    # Extract MMCP content if wrapped
    was_wrapped, content = extract_mmcp_from_wrapped(file_path)
    if was_wrapped:
        print(f"Extracted MMCP content from wrapped file: {file_path}")
    
    # Validate content
    valid, invalid_lines = validate_with_megalithic_regex(content)
    
    # Generate report
    report = generate_proof_report(file_path, valid, invalid_lines)
    
    # Output report
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
        output_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f"Report written to {output_path}")
    else:
        print(json.dumps(report, indent=2))
    
    # Exit with appropriate status code
    if not valid:
        print(f"Validation failed with {len(invalid_lines)} invalid lines.")
        sys.exit(1)
    else:
        print("Validation successful!")
        sys.exit(0)

if __name__ == "__main__":
    main()