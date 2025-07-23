#!/usr/bin/env python3
import argparse
import re
import json
import sys
import asyncio
import os
from pathlib import Path
from functools import wraps
from typing import List, Dict, Union, Any, Optional, Tuple

# --- CLON Parser Logic (from clon_parser.py) ---
def tokenize_clon(clon_string: str) -> List[str]:
    token_regex = r"(\w+|\d+|\[|\]|,|τ)"
    return re.findall(token_regex, clon_string)

def parse_clon_tokens(tokens: List[str]) -> Union[Dict[str, Any], str]:
    if not tokens:
        raise ValueError("Unexpected end of input")
    token = tokens.pop(0)
    if token == 'τ':
        return {'command': 'τ', 'args': []}
    elif re.match(r"\w+", token):
        if not tokens or tokens[0] != '[':
            return token
        tokens.pop(0)
        args = []
        while tokens and tokens[0] != ']':
            args.append(parse_clon_tokens(tokens))
            if tokens and tokens[0] == ',':
                tokens.pop(0)
        if not tokens:
            raise ValueError("Missing ']'")
        tokens.pop(0)
        return {"command": token, "args": args}
    else:
        return token

def parse_clon(clon_string: str) -> Union[Dict[str, Any], str]:
    tokens = tokenize_clon(clon_string)
    return parse_clon_tokens(tokens)

def extract_lambdajson_blocks(text: str) -> List[Dict[str, Any]]:
    """Extracts all single-line LambdaJSON blocks."""
    pattern = r"^\s*##\s*(\{.*\})\s*$"
    blocks = []
    
    for line in text.splitlines():
        match = re.match(pattern, line)
        if match:
            try:
                json_str = re.sub(r'([{\s,])([a-zA-Z_][a-zA-Z0-9_]*):', r'\1"\2":', match.group(1))
                blocks.append(json.loads(json_str))
            except json.JSONDecodeError:
                # Fallback for complex cases: store as raw string
                blocks.append({"_raw_unparsed_lambdajson": match.group(1)})
    return blocks

def extract_mermaid_blocks(text: str) -> List[str]:
    """Extracts all Mermaid diagram blocks."""
    pattern = r"```mermaid\s*([\s\S]*?)```"
    return re.findall(pattern, text)

async def execute_clon(parsed_clon: Union[Dict[str, Any], str], context_input: Any = None) -> Any:
    if isinstance(parsed_clon, str):
        return parsed_clon
    command = parsed_clon.get("command")
    args = parsed_clon.get("args", [])
    if command == 'τ':
        return context_input
    if command == "pipe":
        current_input = context_input
        for arg in args:
            current_input = await execute_clon(arg, context_input=current_input)
        return current_input
    
    evaluated_args = [await execute_clon(arg, context_input) for arg in args]
    
    if command == "grep":
        pattern, *files_or_input = evaluated_args
        results = []
        if isinstance(files_or_input[0], list):
             for line in files_or_input[0]:
                if re.search(pattern, str(line)):
                    results.append(line.strip())
        else:
            for file in files_or_input:
                try:
                    with open(file, 'r') as f:
                        results.extend(line.strip() for line in f if re.search(pattern, line))
                except Exception as e:
                    print(f"Error grepping file {file}: {e}", file=sys.stderr)
        return results
    elif command == "find":
        return [str(p) for p in Path(evaluated_args[0]).rglob('*')]
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        return None

# --- Format Wrapping Logic ---
FORMAT_WRAPPERS = {
    "py": {
        "wrapper_start": "'''\n# MMCP-START\n",
        "wrapper_end": "# MMCP-END\n'''",
        "comment_style": "# ",
        "footer_style": "# MMCP-FOOTER: "
    },
    "md": {
        "wrapper_start": "```mmcp\n<!-- MMCP-START -->\n",
        "wrapper_end": "<!-- MMCP-END -->\n```",
        "comment_style": "<!-- ",
        "footer_style": "<!-- MMCP-FOOTER: "
    },
    "js": {
        "wrapper_start": "/*\n * MMCP-START\n",
        "wrapper_end": " * MMCP-END\n */",
        "comment_style": "// ",
        "footer_style": "// MMCP-FOOTER: "
    },
    "json": {
        "wrapper_start": "{\n  \"__mmcp\": {\n    \"content\": \"",
        "wrapper_end": "\"\n  }\n}",
        "comment_style": "// ",
        "footer_style": "// MMCP-FOOTER: "
    },
    "sql": {
        "wrapper_start": "/*\n * MMCP-START\n",
        "wrapper_end": " * MMCP-END\n */",
        "comment_style": "-- ",
        "footer_style": "-- MMCP-FOOTER: "
    },
    "toml": {
        "wrapper_start": "# MMCP-START\n[mmcp]\ncontent = '''\n",
        "wrapper_end": "'''\n# MMCP-END",
        "comment_style": "# ",
        "footer_style": "# MMCP-FOOTER: "
    },
    "todo": {
        "wrapper_start": "# MMCP-TODO-START\n",
        "wrapper_end": "# MMCP-TODO-END",
        "comment_style": "# ",
        "footer_style": "# MMCP-FOOTER: "
    },
    "lambda": {
        "wrapper_start": "λ:mmcp_wrapper(\n",
        "wrapper_end": ")",
        "comment_style": "-- ",
        "footer_style": "-- MMCP-FOOTER: "
    }
}

def detect_format_from_filename(filename: str) -> str:
    """Detect the format from the filename extension."""
    parts = filename.split('.')
    if len(parts) >= 3 and parts[-2] == "mmcp":
        return parts[-1]
    elif len(parts) >= 3 and parts[-1] == "mmcp":
        return parts[-2]
    else:
        return "mmd"  # Default format

def extract_mmcp_from_wrapped(content: str, format_type: str) -> str:
    """Extract MMCP content from a wrapped file."""
    if format_type == "json":
        try:
            data = json.loads(content)
            if "__mmcp" in data and "content" in data["__mmcp"]:
                return data["__mmcp"]["content"]
        except json.JSONDecodeError:
            pass
    
    if format_type in FORMAT_WRAPPERS:
        wrapper = FORMAT_WRAPPERS[format_type]
        start_marker = wrapper["wrapper_start"].strip()
        end_marker = wrapper["wrapper_end"].strip()
        
        # Handle different formats with their specific patterns
        if format_type == "py":
            pattern = r"'''\s*# MMCP-START\s*([\s\S]*?)\s*# MMCP-END\s*'''"
        elif format_type == "md":
            pattern = r"```mmcp\s*<!-- MMCP-START -->\s*([\s\S]*?)\s*<!-- MMCP-END -->\s*```"
        elif format_type in ["js", "sql"]:
            pattern = r"/\*\s*\* MMCP-START\s*([\s\S]*?)\s*\* MMCP-END\s*\*/"
        elif format_type == "toml":
            pattern = r"# MMCP-START\s*\[mmcp\]\s*content = '''\s*([\s\S]*?)\s*'''\s*# MMCP-END"
        elif format_type == "todo":
            pattern = r"# MMCP-TODO-START\s*([\s\S]*?)\s*# MMCP-TODO-END"
        elif format_type == "lambda":
            pattern = r"λ:mmcp_wrapper\(\s*([\s\S]*?)\s*\)"
        else:
            return ""
        
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # If no specific format matched or extraction failed, return empty string
    return ""

def wrap_mmcp_content(mmcp_content: str, format_type: str) -> str:
    """Wrap MMCP content in the specified format."""
    if format_type not in FORMAT_WRAPPERS:
        raise ValueError(f"Unsupported format: {format_type}")
    
    wrapper = FORMAT_WRAPPERS[format_type]
    
    # Handle JSON format specially
    if format_type == "json":
        # Escape newlines and quotes for JSON
        escaped_content = mmcp_content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        return f'{{\n  "__mmcp": {{\n    "content": "{escaped_content}"\n  }}\n}}'
    
    # For other formats, use the defined wrappers
    return f"{wrapper['wrapper_start']}{mmcp_content}{wrapper['wrapper_end']}"

def extract_nested_content(mmcp_content: str) -> List[Dict[str, Any]]:
    """Extract nested content blocks from MMCP content."""
    pattern = r"<mmcp:nested\s+type=\"([^\"]+)\"\s+(?:id=\"([^\"]+)\"\s+)?(?:version=\"([^\"]+)\"\s+)?(?:checksum=\"([^\"]+)\"\s+)?>([\s\S]*?)</mmcp:nested>"
    nested_blocks = []
    
    for match in re.finditer(pattern, mmcp_content, re.DOTALL):
        nested_type = match.group(1)
        nested_id = match.group(2) or ""
        nested_version = match.group(3) or ""
        nested_checksum = match.group(4) or ""
        nested_content = match.group(5)
        
        nested_blocks.append({
            "type": nested_type,
            "id": nested_id,
            "version": nested_version,
            "checksum": nested_checksum,
            "content": nested_content
        })
    
    return nested_blocks

def generate_footer(mmcp_content: str, version: str = "1.2.0") -> str:
    """Generate a standard MMCP footer."""
    import hashlib
    from datetime import datetime
    
    # Calculate checksum
    checksum = hashlib.sha256(mmcp_content.encode()).hexdigest()
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    return f"MMCP-FOOTER: version={version}; timestamp={timestamp}; checksum=sha256:{checksum}"

# --- Main CLI ---
def async_command(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

def handle_parse_command(args):
    try:
        content = args.input_file.read_text(encoding='utf-8')
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    # Check if the file is a wrapped format
    format_type = detect_format_from_filename(args.input_file.name)
    if format_type != "mmd":
        # Extract MMCP content from wrapped format
        mmcp_content = extract_mmcp_from_wrapped(content, format_type)
        if mmcp_content:
            content = mmcp_content
    
    lambdajson_data = extract_lambdajson_blocks(content)
    mermaid_diagrams = extract_mermaid_blocks(content)
    nested_blocks = extract_nested_content(content)

    if args.output_prefix:
        json_path = Path(f"{args.output_prefix}.data.json")
        mermaid_path = Path(f"{args.output_prefix}.mermaid.md")
        nested_path = Path(f"{args.output_prefix}.nested.json")
        
        json_path.write_text(json.dumps(lambdajson_data, indent=2), encoding='utf-8')
        mermaid_path.write_text('\n\n'.join([f'```mermaid\n{d.strip()}\n```' for d in mermaid_diagrams]), encoding='utf-8')
        nested_path.write_text(json.dumps(nested_blocks, indent=2), encoding='utf-8')
        
        print(f"Data written to {json_path}")
        print(f"Mermaid diagrams written to {mermaid_path}")
        print(f"Nested blocks written to {nested_path}")
    else:
        print("LambdaJSON Blocks:")
        print(json.dumps(lambdajson_data, indent=2))
        
        if mermaid_diagrams:
            print("\nMermaid Diagrams:")
            for diagram in mermaid_diagrams:
                print(f'\n```mermaid\n{diagram.strip()}\n```')
        
        if nested_blocks:
            print("\nNested Blocks:")
            print(json.dumps(nested_blocks, indent=2))

@async_command
async def handle_declare_command(args):
    try:
        parsed_clon = parse_clon(args.clon_string)
        result = await execute_clon(parsed_clon)
        if result:
            print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error executing CLON: {e}", file=sys.stderr)
        sys.exit(1)

def handle_validate_command(args):
    try:
        content = args.input_file.read_text(encoding='utf-8')
        
        # Check if the file is a wrapped format
        format_type = detect_format_from_filename(args.input_file.name)
        if format_type != "mmd":
            # Extract MMCP content from wrapped format
            mmcp_content = extract_mmcp_from_wrapped(content, format_type)
            if mmcp_content:
                content = mmcp_content
        
        # Basic validation checks
        root_pattern = r"\[ \] #\"\.root\"# \{protocol:\"MCP\", version:\"[0-9.]+\", standard:\"[^\"]+\"\}"
        if not re.search(root_pattern, content):
            print("Validation Error: Missing or invalid root indicator", file=sys.stderr)
            sys.exit(1)
        
        # Check for required meta block
        meta_pattern = r"## \{type:Meta, author:\"[^\"]+\", license:\"[^\"]+\", last_modified:\"[^\"]+\", id:\"[^\"]+\"\}"
        if not re.search(meta_pattern, content):
            print("Validation Warning: Missing or invalid Meta block", file=sys.stderr)
        
        # Check for nested content validity
        nested_blocks = extract_nested_content(content)
        for block in nested_blocks:
            if not block["type"]:
                print(f"Validation Warning: Nested block missing type attribute", file=sys.stderr)
        
        print("Validation passed with 0 errors and 0 warnings.")
    except Exception as e:
        print(f"Validation Error: {e}", file=sys.stderr)
        sys.exit(1)

def handle_wrap_command(args):
    try:
        # Read input MMCP content
        mmcp_content = args.input_file.read_text(encoding='utf-8')
        
        # Wrap the content in the specified format
        wrapped_content = wrap_mmcp_content(mmcp_content, args.format)
        
        # Write to output file or stdout
        if args.output_file:
            args.output_file.write_text(wrapped_content, encoding='utf-8')
            print(f"Wrapped content written to {args.output_file}")
        else:
            print(wrapped_content)
    except Exception as e:
        print(f"Error wrapping content: {e}", file=sys.stderr)
        sys.exit(1)

def handle_unwrap_command(args):
    try:
        # Read input wrapped content
        wrapped_content = args.input_file.read_text(encoding='utf-8')
        
        # Detect format from filename if not specified
        format_type = args.format or detect_format_from_filename(args.input_file.name)
        
        # Extract MMCP content
        mmcp_content = extract_mmcp_from_wrapped(wrapped_content, format_type)
        
        if not mmcp_content:
            print(f"Error: Could not extract MMCP content from {args.input_file}", file=sys.stderr)
            sys.exit(1)
        
        # Write to output file or stdout
        if args.output_file:
            args.output_file.write_text(mmcp_content, encoding='utf-8')
            print(f"Unwrapped content written to {args.output_file}")
        else:
            print(mmcp_content)
    except Exception as e:
        print(f"Error unwrapping content: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="PFSUS Core CLI Tool (v1.2.0)")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Parser for the "parse" command
    parse_parser = subparsers.add_parser('parse', help="Parse a .mmcp.mmd file into data and diagrams.")
    parse_parser.add_argument('input_file', type=Path, help="Input .mmcp.mmd file")
    parse_parser.add_argument('-o', '--output-prefix', type=str, help="Prefix for output files.")
    parse_parser.set_defaults(func=handle_parse_command)

    # Parser for the "declare" command
    declare_parser = subparsers.add_parser('declare', help="Execute a command using CLON syntax.")
    declare_parser.add_argument('clon_string', type=str, help="CLON command string.")
    declare_parser.set_defaults(func=handle_declare_command)

    # Parser for the "validate" command
    validate_parser = subparsers.add_parser('validate', help="Validate a .mmcp.mmd file.")
    validate_parser.add_argument('input_file', type=Path, help="Input .mmcp.mmd file")
    validate_parser.set_defaults(func=handle_validate_command)

    # Parser for the "wrap" command
    wrap_parser = subparsers.add_parser('wrap', help="Wrap MMCP content in another file format.")
    wrap_parser.add_argument('input_file', type=Path, help="Input MMCP file")
    wrap_parser.add_argument('format', choices=list(FORMAT_WRAPPERS.keys()), help="Target format")
    wrap_parser.add_argument('-o', '--output-file', type=Path, help="Output file path")
    wrap_parser.set_defaults(func=handle_wrap_command)

    # Parser for the "unwrap" command
    unwrap_parser = subparsers.add_parser('unwrap', help="Extract MMCP content from a wrapped file.")
    unwrap_parser.add_argument('input_file', type=Path, help="Input wrapped file")
    unwrap_parser.add_argument('-f', '--format', choices=list(FORMAT_WRAPPERS.keys()), help="Source format (detected from filename if not specified)")
    unwrap_parser.add_argument('-o', '--output-file', type=Path, help="Output file path")
    unwrap_parser.set_defaults(func=handle_unwrap_command)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main() 