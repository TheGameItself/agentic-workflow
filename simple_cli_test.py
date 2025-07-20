#!/usr/bin/env python3
"""
Simple CLI test without complex imports
"""

import click
import sys
import os
from pathlib import Path

@click.group()
def cli():
    """Simple MCP CLI Test"""
    pass

@cli.command()
@click.option('--pattern', prompt='Grep pattern', help='Pattern to search for')
@click.option('--path', default='.', help='Path to search in')
def grep_search(pattern, path):
    """Search for patterns in files using grep-like functionality."""
    import re
    from pathlib import Path
    
    path_obj = Path(path)
    if not path_obj.exists():
        click.echo(f"Path does not exist: {path}")
        return
    
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        click.echo(f"Invalid regex pattern: {e}")
        return
    
    results = []
    for md_file in path_obj.rglob("*.md"):
        try:
            with open(md_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines, 1):
                if regex.search(line):
                    results.append({
                        'file': str(md_file),
                        'line': i,
                        'content': line.strip()
                    })
                    if len(results) >= 10:  # Limit results
                        break
        except Exception as e:
            click.echo(f"Error reading {md_file}: {e}")
    
    click.echo(f"Found {len(results)} matches:")
    for result in results:
        click.echo(f"{result['file']}:{result['line']}: {result['content']}")

@cli.command()
@click.option('--query', prompt='EARS search query', help='EARS format query')
def ears_search(query):
    """Search using EARS format queries."""
    import re
    
    # Parse EARS query: [Event] -> [Action] -> [Result] -> [System]
    ears_pattern = re.compile(r'\[([^\]]+)\]\s*->\s*\[([^\]]+)\]\s*->\s*\[([^\]]+)\]\s*->\s*\[([^\]]+)\]')
    match = ears_pattern.match(query)
    
    if not match:
        click.echo("Invalid EARS format. Use: [Event] -> [Action] -> [Result] -> [System]")
        return
    
    event, action, result, system = match.groups()
    click.echo(f"EARS Query parsed:")
    click.echo(f"  Event: {event}")
    click.echo(f"  Action: {action}")
    click.echo(f"  Result: {result}")
    click.echo(f"  System: {system}")
    
    # Simple search in docs
    docs_path = Path('docs')
    if docs_path.exists():
        found_files = []
        for md_file in docs_path.rglob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if (event.lower() in content.lower() or 
                    action.lower() in content.lower() or 
                    result.lower() in content.lower() or 
                    system.lower() in content.lower()):
                    found_files.append(str(md_file))
            except Exception as e:
                click.echo(f"Error reading {md_file}: {e}")
        
        click.echo(f"\nFound {len(found_files)} matching files:")
        for file in found_files[:5]:  # Show first 5
            click.echo(f"  - {file}")

if __name__ == '__main__':
    cli() 