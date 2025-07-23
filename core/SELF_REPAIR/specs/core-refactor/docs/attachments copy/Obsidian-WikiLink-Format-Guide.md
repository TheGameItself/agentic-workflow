---
tags: [documentation, obsidian, wikilinks, format, guide]
graph-view-group: Documentation
---

# Obsidian WikiLink Format Guide

## Epic
**As a** documentation maintainer
**I want** to understand and use proper Obsidian WikiLink format
**So that** I can create consistent, navigable documentation

## Overview

Obsidian WikiLinks are the primary method for creating internal links between documents in the MCP Agentic Workflow documentation. This guide covers the syntax, best practices, and patterns used throughout the project.

## 🔗 WikiLink Syntax

### Basic Format
```markdown
[[target-file]]
```

### With Display Text
```markdown
[[target-file|Display Text]]
```

### With Section Headers
```markdown
[[target-file#section-name]]
[[target-file#section-name|Display Text]]
```

### Relative Paths
```markdown
[[../parent-directory/file]]
[[../parent-directory/file|Display Text]]
```

## 📁 Canonical Path Structure

### Directory-Based Links
The project uses a canonical directory structure with lowercase names:

```markdown
# Correct patterns
[[api/README|API Documentation]]
[[architecture/README|System Architecture]]
[[core-systems/README|Core Systems]]
[[development/README|Development]]
[[testing/README|Testing]]
[[troubleshooting/README|Troubleshooting]]
[[performance-optimization/README|Performance]]
[[security/README|Security]]
[[deployment/README|Deployment]]
[[genetic-system/README|Genetic System]]
[[hormone-system/README|Hormone System]]
[[pattern-recognition/README|Pattern Recognition]]
[[p2p-network/README|P2P Network]]
[[workflow/README|Workflow]]
[[configuration/README|Configuration]]
[[guides/README|Guides]]
[[attachments/README|Attachments]]
```

### Relative Path Examples
```markdown
# From subdirectory to parent
[[../README|Project Overview]]

# From subdirectory to sibling
[[../api/README|API Documentation]]
[[../architecture/README|Architecture]]

# From subdirectory to other subdirectory
[[../core-systems/README|Core Systems]]
[[../troubleshooting/README|Troubleshooting]]
```

## 🎯 Best Practices

### 1. Use Canonical Paths
- **Always use lowercase directory names**
- **Reference README files for directory overviews**
- **Use relative paths when linking from subdirectories**

### 2. Consistent Naming
```markdown
# Good - Clear and descriptive
[[api/README|API Documentation]]
[[architecture/README|System Architecture]]

# Avoid - Vague or inconsistent
[[API|API]]
[[Architecture|Architecture]]
```

### 3. Cross-Reference Patterns
```markdown
# Standard cross-reference format
## Related Documentation
- [[../architecture/README|System Architecture]] - System design overview
- [[../core-systems/README|Core Systems]] - Core system components
- [[../api/README|API Reference]] - Technical API documentation
```

### 4. Navigation Links
```markdown
# Main navigation section
## Quick Links
- [[getting-started/README|Getting Started]] - Installation and setup
- [[development/README|Development]] - Developer guides
- [[troubleshooting/README|Troubleshooting]] - Common issues
```

## 🔧 Common Patterns in This Project

### 1. README References
```markdown
# Directory overview links
- [[README]] - Category overview
- [[../api/README|API]] - API documentation
- [[../development/README|Development]] - Development guides
```

### 2. Cross-System References
```markdown
# System component links
- [[../architecture/README|System-Architecture]] - System architecture
- [[../core-systems/README|Core-Systems]] - Core system components
- [[../troubleshooting/README|Troubleshooting]] - Troubleshooting guide
```

### 3. Documentation Index Links
```markdown
# Main documentation references
- [[Documentation-Index.md]] - Main documentation index
- [[README.md]] - Project overview
```

## ❌ Common Mistakes to Avoid

### 1. Title-Case Directory Names
```markdown
# Wrong - Don't use Title-Case
[[API/README|API]]
[[Architecture/README|Architecture]]
[[Core-Systems/README|Core Systems]]

# Correct - Use lowercase
[[api/README|API]]
[[architecture/README|Architecture]]
[[core-systems/README|Core Systems]]
```

### 2. Missing README References
```markdown
# Wrong - Direct file reference without README
[[api/README|API]]
[[architecture/README|Architecture]]

# Correct - Reference README for directory overview
[[api/README|API]]
[[architecture/README|Architecture]]
```

### 3. Inconsistent Display Text
```markdown
# Wrong - Inconsistent naming
[[api/README|API]]
[[architecture/README|Architecture]]
[[core-systems/README|Core-Systems]]

# Correct - Consistent naming
[[api/README|API]]
[[architecture/README|Architecture]]
[[core-systems/README|Core Systems]]
```

## 🔍 Validation and Testing

### 1. Check for Broken Links
```bash
# Find broken WikiLinks
grep -r "\[\[.*\|.*(missing)\]\]" docs/ --include="*.md"
```

### 2. Validate Canonical Structure
```bash
# Check for Title-Case directory references
grep -r "\[\[[A-Z]" docs/ --include="*.md"
```

### 3. Verify Cross-References
```bash
# Find all WikiLinks
grep -r "\[\[.*\]\]" docs/ --include="*.md"
```

## 📊 Obsidian Graph Integration

### Color Groups
The project uses organized color groups in `.obsidian/graph.json`:

```json
{
  "colorGroups": [
    {
      "query": "path:canonical/api",
      "color": { "a": 1, "rgb": 65280 }
    },
    {
      "query": "path:canonical/architecture",
      "color": { "a": 1, "rgb": 16711935 }
    }
  ]
}
```

### Tags for Organization
```markdown
---
tags: [documentation, api, obsidian, guide]
graph-view-group: API
---
```

## 🚀 Automation and Maintenance

### 1. Automated Link Fixing
The project includes scripts to automatically fix broken WikiLinks:

```python
# Fix common broken link patterns
def fix_broken_links():
    patterns = [
        (r'\[\[Core-Systems\|Core-Systems \(missing\)\]\]', '[[../core-systems/README|Core-Systems]]'),
        (r'\[\[API\|API \(missing\)\]\]', '[[../api/README|API]]'),
        (r'\[\[Architecture\|Architecture \(missing\)\]\]', '[[../architecture/README|Architecture]]')
    ]
```

### 2. Canonical Structure Enforcement
```python
# Ensure all links use canonical paths
def enforce_canonical_paths():
    canonical_dirs = [
        'api', 'architecture', 'core-systems', 'development',
        'testing', 'troubleshooting', 'performance-optimization',
        'security', 'deployment', 'genetic-system', 'hormone-system',
        'pattern-recognition', 'p2p-network', 'workflow'
    ]
```

## 📚 Related Documentation

- [[Documentation-Index.md]] - Main documentation index
- [[README.md]] - Project overview
- [[MCP-Self-Improvement-Guide.md]] - How MCP tools improve documentation
- [[Documentation-Status-Report.md]] - Overall documentation status

## 🎯 Summary

### Key Principles
1. **Use canonical lowercase paths** for all directory references
2. **Reference README files** for directory overviews
3. **Use relative paths** when linking from subdirectories
4. **Maintain consistent display text** across all links
5. **Validate links regularly** to prevent broken references

### Benefits
- **Consistent Navigation**: All links follow the same pattern
- **Easy Maintenance**: Canonical structure simplifies updates
- **Graph Visualization**: Proper organization in Obsidian graph view
- **Cross-Reference Discovery**: Users can easily explore related content

---

*This guide ensures consistent WikiLink usage throughout the MCP Agentic Workflow documentation system.* 