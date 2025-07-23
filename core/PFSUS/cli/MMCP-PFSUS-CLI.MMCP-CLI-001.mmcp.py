#!/usr/bin/env python3
'''
# MMCP-START
[ ] #".root"# {protocol:"MCP", version:"1.2.1", standard:"PFSUS+EARS+LambdaJSON"}
# MMCP-PFSUS Command Line Interface
## {type:Meta, author:"Kalxi", license:"MIT", last_modified:"2025-07-21T00:00:00Z", id:"MMCP.CLI.MMCP-CLI-001"}
## {type:Schema, $schema:"https://json-schema.org/draft/2020-12/schema", required:["type","id","version"], properties:{type:{type:"string"},id:{type:"string"},version:{type:"string"},last_modified:{type:"string",format:"date-time"},author:{type:"string"}}}
## {type:Changelog, entries:[
  {"2025-07-21":"Updated CLI tool with format wrapping support and new naming convention."},
  {"2025-07-21":"Added support for megalithic regex validation and proof-counterproof."}
]}

## {type:Tool, id:"MMCP-CLI-001", desc:"Command line interface for working with MMCP documents and formats."}
# MMCP-END
'''

import argparse
import re
import json
import sys
import asyncio
import os
import hashlib
from pathlib import Path
from functools import wraps
from typing import List, Dict, Union, Any, Optional, Tuple
from datetime import datetime

# --- CLON Parser Logic ---
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
    },
    "alef": {
        "wrapper_start": "ℵ:mmcp_wrapper(\n",
        "wrapper_end": ")",
        "comment_style": "-- ",
        "footer_style": "-- MMCP-FOOTER: "
    },
    "delta": {
        "wrapper_start": "Δ:mmcp_wrapper(\n",
        "wrapper_end": ")",
        "comment_style": "-- ",
        "footer_style": "-- MMCP-FOOTER: "
    },
    "beta": {
        "wrapper_start": "β:mmcp_wrapper(\n",
        "wrapper_end": ")",
        "comment_style": "-- ",
        "footer_style": "-- MMCP-FOOTER: "
    },
    "omega": {
        "wrapper_start": "Ω:mmcp_wrapper(\n",
        "wrapper_end": ")",
        "comment_style": "-- ",
        "footer_style": "-- MMCP-FOOTER: "
    },
    "imaginary": {
        "wrapper_start": "i:mmcp_wrapper(\n",
        "wrapper_end": ")",
        "comment_style": "-- ",
        "footer_style": "-- MMCP-FOOTER: "
    },
    "turing": {
        "wrapper_start": "τ:mmcp_wrapper(\n",
        "wrapper_end": ")",
        "comment_style": "-- ",
        "footer_style": "-- MMCP-FOOTER: "
    },
    "base64": {
        "wrapper_start": "base64:mmcp_wrapper(\n",
        "wrapper_end": ")",
        "comment_style": "# ",
        "footer_style": "# MMCP-FOOTER: "
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

def generate_footer(mmcp_content: str, version: str = "1.2.1") -> str:
    """Generate a standard MMCP footer."""
    # Calculate checksum
    checksum = hashlib.sha256(mmcp_content.encode()).hexdigest()
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    return f"MMCP-FOOTER: version={version}; timestamp={timestamp}; checksum=sha256:{checksum}"

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
        
        # Validate with megalithic regex
        valid, invalid_lines = validate_with_megalithic_regex(content)
        if not valid:
            print(f"Validation Error: Found {len(invalid_lines)} invalid lines:", file=sys.stderr)
            for line in invalid_lines[:10]:  # Show first 10 invalid lines
                print(f"  {line}", file=sys.stderr)
            if len(invalid_lines) > 10:
                print(f"  ... and {len(invalid_lines) - 10} more.", file=sys.stderr)
            sys.exit(1)
        
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

def handle_proof_command(args):
    try:
        # Read input content
        content = args.input_file.read_text(encoding='utf-8')
        
        # Check if the file is a wrapped format
        format_type = detect_format_from_filename(args.input_file.name)
        if format_type != "mmd":
            # Extract MMCP content from wrapped format
            mmcp_content = extract_mmcp_from_wrapped(content, format_type)
            if mmcp_content:
                content = mmcp_content
        
        # Validate with megalithic regex
        valid, invalid_lines = validate_with_megalithic_regex(content)
        
        # Generate report
        report = {
            "file": str(args.input_file),
            "valid": valid,
            "timestamp": datetime.now().isoformat(),
            "invalid_lines": invalid_lines,
            "line_count": len(content.splitlines()),
            "valid_percentage": 100 if valid else (1 - len(invalid_lines) / len(content.splitlines())) * 100
        }
        
        # Output report
        if args.output_file:
            args.output_file.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(f"Proof report written to {args.output_file}")
        else:
            print(json.dumps(report, indent=2))
        
        if not valid:
            sys.exit(1)
    except Exception as e:
        print(f"Error generating proof: {e}", file=sys.stderr)
        sys.exit(1)

def handle_encode_base64_command(args):
    try:
        # Read input content
        content = args.input_file.read_text(encoding='utf-8')
        
        # Encode to base64
        import base64
        encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
        
        # Wrap in base64 format if requested
        if args.wrap:
            encoded_content = f"base64:mmcp_wrapper(\n  {encoded_content}\n)"
        
        # Write to output file or stdout
        if args.output_file:
            args.output_file.write_text(encoded_content, encoding='utf-8')
            print(f"Base64 encoded content written to {args.output_file}")
        else:
            print(encoded_content)
    except Exception as e:
        print(f"Error encoding content: {e}", file=sys.stderr)
        sys.exit(1)

def handle_decode_base64_command(args):
    try:
        # Read input content
        content = args.input_file.read_text(encoding='utf-8')
        
        # Check if content is wrapped in base64 format
        base64_pattern = r"base64:mmcp_wrapper\(\s*(.*?)\s*\)"
        match = re.search(base64_pattern, content, re.DOTALL)
        if match:
            content = match.group(1).strip()
        
        # Decode from base64
        import base64
        try:
            decoded_content = base64.b64decode(content).decode('utf-8')
        except Exception as e:
            print(f"Error decoding base64 content: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Write to output file or stdout
        if args.output_file:
            args.output_file.write_text(decoded_content, encoding='utf-8')
            print(f"Decoded content written to {args.output_file}")
        else:
            print(decoded_content)
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        sys.exit(1)

def handle_convert_command(args):
    try:
        # Read input content
        content = args.input_file.read_text(encoding='utf-8')
        
        # Detect source format from filename if not specified
        source_format = args.source_format or detect_format_from_filename(args.input_file.name)
        
        # Extract MMCP content if wrapped
        if source_format != "mmd":
            mmcp_content = extract_mmcp_from_wrapped(content, source_format)
            if mmcp_content:
                content = mmcp_content
        
        # Wrap in target format
        wrapped_content = wrap_mmcp_content(content, args.target_format)
        
        # Write to output file or stdout
        if args.output_file:
            args.output_file.write_text(wrapped_content, encoding='utf-8')
            print(f"Converted content written to {args.output_file}")
        else:
            print(wrapped_content)
    except Exception as e:
        print(f"Error converting content: {e}", file=sys.stderr)
        sys.exit(1)

def handle_generate_command(args):
    try:
        # Get template path
        template_dir = Path(__file__).parent.parent / ".legend.mmcp" / "templates"
        template_path = template_dir / f"{args.template}.mmcp.md"
        
        if not template_path.exists():
            print(f"Error: Template '{args.template}' not found", file=sys.stderr)
            sys.exit(1)
        
        # Read template
        template_content = template_path.read_text(encoding='utf-8')
        
        # Extract MMCP content if wrapped
        template_content = extract_mmcp_from_wrapped(template_content, "md")
        
        # Get schema path if specified
        schema_content = None
        if args.schema:
            schema_path = Path(__file__).parent.parent / ".legend.mmcp" / "schemas" / f"{args.schema}.schema.json"
            if schema_path.exists():
                schema_content = schema_path.read_text(encoding='utf-8')
        
        # Generate unique ID if not provided
        if not args.id:
            import uuid
            args.id = f"MMCP-{uuid.uuid4().hex[:8].upper()}"
        
        # Replace placeholders
        from datetime import datetime
        replacements = {
            "{TITLE}": args.title or "Untitled Document",
            "{VERSION}": args.version or "1.0.0",
            "{AUTHOR}": args.author or "Kalxi",
            "{DATE}": datetime.now().strftime("%Y-%m-%d"),
            "{ID}": args.id,
            "{TYPE}": args.type or "Document",
            "{DESCRIPTION}": args.description or f"Generated {args.template} document",
            "{INTRODUCTION}": args.introduction or "This document was automatically generated.",
            "{SECTION_TITLE}": "Content",
            "{SECTION_CONTENT}": "Add your content here.",
            "{FILENAME}": str(args.output_file) if args.output_file else f"{args.title or 'document'}.mmcp.md",
            "{ADDRESS}": args.id.lower(),
            "{CHECKSUM}": "0" * 64,  # Placeholder, will be updated later
            "{STRUCTURE}": "content, sections"
        }
        
        for key, value in replacements.items():
            template_content = template_content.replace(key, value)
        
        # Calculate actual checksum
        import hashlib
        checksum = hashlib.sha256(template_content.encode()).hexdigest()
        template_content = template_content.replace("0" * 64, checksum)
        
        # Wrap in calculus notation if specified
        if args.calculus:
            if args.calculus in FORMAT_WRAPPERS:
                wrapper = FORMAT_WRAPPERS[args.calculus]
                template_content = f"{wrapper['wrapper_start'].strip()}\n{template_content}\n{wrapper['wrapper_end'].strip()}"
        
        # Write to output file or stdout
        if args.output_file:
            # Ensure directory exists
            args.output_file.parent.mkdir(parents=True, exist_ok=True)
            args.output_file.write_text(template_content, encoding='utf-8')
            print(f"Generated document written to {args.output_file}")
        else:
            print(template_content)
    except Exception as e:
        print(f"Error generating document: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="MMCP-PFSUS CLI Tool (v1.2.1)")
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

    # Parser for the "proof" command
    proof_parser = subparsers.add_parser('proof', help="Validate a file using megalithic regex and generate a proof report.")
    proof_parser.add_argument('input_file', type=Path, help="Input file to validate")
    proof_parser.add_argument('-o', '--output-file', type=Path, help="Output report file path")
    proof_parser.set_defaults(func=handle_proof_command)
    
    # Parser for the "encode" command
    encode_parser = subparsers.add_parser('encode', help="Encode MMCP content in base64.")
    encode_parser.add_argument('input_file', type=Path, help="Input file to encode")
    encode_parser.add_argument('-o', '--output-file', type=Path, help="Output file path")
    encode_parser.add_argument('-w', '--wrap', action='store_true', help="Wrap the encoded content in base64 format")
    encode_parser.set_defaults(func=handle_encode_base64_command)
    
    # Parser for the "decode" command
    decode_parser = subparsers.add_parser('decode', help="Decode base64-encoded MMCP content.")
    decode_parser.add_argument('input_file', type=Path, help="Input file to decode")
    decode_parser.add_argument('-o', '--output-file', type=Path, help="Output file path")
    decode_parser.set_defaults(func=handle_decode_base64_command)
    
    # Parser for the "convert" command
    convert_parser = subparsers.add_parser('convert', help="Convert between different wrapped formats.")
    convert_parser.add_argument('input_file', type=Path, help="Input file to convert")
    convert_parser.add_argument('target_format', choices=list(FORMAT_WRAPPERS.keys()), help="Target format")
    convert_parser.add_argument('-s', '--source-format', choices=list(FORMAT_WRAPPERS.keys()), help="Source format (detected from filename if not specified)")
    convert_parser.add_argument('-o', '--output-file', type=Path, help="Output file path")
    convert_parser.set_defaults(func=handle_convert_command)
    
    # Parser for the "generate" command
    generate_parser = subparsers.add_parser('generate', help="Generate a new MMCP document from a template.")
    generate_parser.add_argument('template', help="Template name (e.g., standard, spec, agent)")
    generate_parser.add_argument('-o', '--output-file', type=Path, help="Output file path")
    generate_parser.add_argument('-t', '--title', help="Document title")
    generate_parser.add_argument('-v', '--version', help="Document version")
    generate_parser.add_argument('-a', '--author', help="Document author")
    generate_parser.add_argument('-i', '--id', help="Document ID")
    generate_parser.add_argument('-d', '--description', help="Document description")
    generate_parser.add_argument('-n', '--introduction', help="Document introduction")
    generate_parser.add_argument('-y', '--type', help="Document type")
    generate_parser.add_argument('-s', '--schema', help="Schema name")
    generate_parser.add_argument('-c', '--calculus', choices=['lambda', 'alef', 'delta', 'beta', 'omega', 'imaginary', 'turing'], 
                               default='lambda', help="Calculus notation wrapper")
    generate_parser.set_defaults(func=handle_generate_command)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main() 