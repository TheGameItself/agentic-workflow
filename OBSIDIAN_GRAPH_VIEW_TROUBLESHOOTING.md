# Obsidian Graph View Troubleshooting Guide

## Overview

This guide addresses common Obsidian graph view issues and provides solutions based on research and best practices.

## 🔍 **Common Graph View Issues**

### **1. Graph View Not Loading**
**Symptoms**: Graph view appears empty or doesn't load at all

**Solutions**:
- **Restart Obsidian**: Close and reopen Obsidian completely
- **Clear Cache**: Delete `.obsidian/cache` folder and restart
- **Check File Permissions**: Ensure Obsidian has read access to the vault
- **Verify Vault Path**: Make sure the vault path is correct and accessible

### **2. Missing Color Groups**
**Symptoms**: Files appear but without proper color coding

**Solutions**:
- **Check Graph Configuration**: Verify `.obsidian/graph.json` exists and is valid JSON
- **Update Path Queries**: Ensure path queries match actual directory structure
- **Restart Graph View**: Toggle graph view off and on in Obsidian

### **3. Files Not Appearing**
**Symptoms**: Some files don't show up in the graph view

**Solutions**:
- **Check File Extensions**: Ensure files have `.md` extension
- **Verify File Location**: Files must be within the vault directory
- **Check Graph Settings**: Ensure "Show Orphans" is enabled
- **Refresh Graph**: Use Ctrl+R (Cmd+R on Mac) to refresh

### **4. Broken Links Causing Issues**
**Symptoms**: Graph view shows broken connections or errors

**Solutions**:
- **Fix WikiLinks**: Resolve all broken `[[WikiLinks]]` in documentation
- **Update References**: Ensure all internal links point to existing files
- **Validate Paths**: Check that all referenced files exist

## 🔧 **Configuration Fixes Applied**

### **1. Updated Graph Configuration**
The graph configuration has been updated to use the correct canonical paths:

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
    // ... all paths updated to canonical structure
  ]
}
```

### **2. Removed Non-Existent Directories**
Removed color groups for directories that don't exist:
- ❌ `path:reference` (doesn't exist)
- ❌ `path:tutorials` (doesn't exist) 
- ❌ `path:examples` (doesn't exist)
- ❌ `path:cli` (doesn't exist)
- ❌ `path:community` (doesn't exist)
- ❌ `path:android` (doesn't exist)
- ❌ `path:llm` (doesn't exist)
- ❌ `path:memory-system` (doesn't exist)
- ❌ `path:simulation-layer` (doesn't exist)
- ❌ `path:release` (doesn't exist)

### **3. Added Missing Directories**
Added color groups for directories that do exist:
- ✅ `path:canonical/testing`
- ✅ `path:canonical/security`
- ✅ `path:canonical/deployment`
- ✅ `path:canonical/workflow`
- ✅ `path:canonical/configuration`
- ✅ `path:canonical/user-guides`

## 🚀 **Step-by-Step Recovery Process**

### **Step 1: Verify Vault Structure**
```bash
# Check if canonical directory exists
ls -la docs/canonical/

# Verify markdown files exist
find docs/canonical -name "*.md" | head -10
```

### **Step 2: Restart Obsidian**
1. Close Obsidian completely
2. Wait 10 seconds
3. Reopen Obsidian
4. Open the vault: `agentic-workflow/docs/`

### **Step 3: Clear Graph Cache**
```bash
# Remove graph cache (if it exists)
rm -rf docs/.obsidian/cache/graph.json
```

### **Step 4: Refresh Graph View**
1. Open Graph View in Obsidian
2. Press `Ctrl+R` (or `Cmd+R` on Mac) to refresh
3. Wait for graph to rebuild

### **Step 5: Verify Configuration**
Check that `.obsidian/config.json` contains:
```json
{
  "graph": {
    "enabled": true,
    "showTags": true,
    "showAttachments": true,
    "showOrphans": true,
    "local": true
  }
}
```

## 📊 **Expected Graph View Behavior**

### **When Working Correctly**:
- **Color Groups**: 20+ organized color groups for different documentation areas
- **File Nodes**: All markdown files appear as nodes
- **Connections**: WikiLinks create visible connections between files
- **Tags**: Files are colored based on their tags and directory location
- **Orphans**: Files without links appear as isolated nodes

### **Color Group Organization**:
- **Green**: API documentation (`path:canonical/api`)
- **Purple**: Architecture (`path:canonical/architecture`)
- **Orange**: Core Systems (`path:canonical/core-systems`)
- **Blue**: Development (`path:canonical/development`)
- **Red**: Security (`path:canonical/security`)
- **Gray**: Performance (`path:canonical/performance-optimization`)

## 🔍 **Diagnostic Commands**

### **Check File Structure**
```bash
# List all markdown files
find docs/canonical -name "*.md" | wc -l

# Check for broken WikiLinks
grep -r "\[\[.*\|.*(missing)\]\]" docs/canonical/ --include="*.md"

# Verify graph configuration
cat docs/.obsidian/graph.json | jq '.colorGroups | length'
```

### **Validate Configuration**
```bash
# Check if JSON is valid
cat docs/.obsidian/graph.json | jq '.'

# Count color groups
cat docs/.obsidian/graph.json | jq '.colorGroups | length'

# List all path queries
cat docs/.obsidian/graph.json | jq '.colorGroups[].query'
```

## 🛠️ **Advanced Troubleshooting**

### **If Graph Still Doesn't Work**:

1. **Check Obsidian Version**: Ensure you're using Obsidian 1.0+
2. **Disable Plugins**: Temporarily disable all community plugins
3. **Test with Minimal Vault**: Create a test vault with just a few files
4. **Check System Resources**: Ensure sufficient RAM and CPU available
5. **Reinstall Obsidian**: As a last resort, reinstall Obsidian

### **Performance Optimization**:
- **Limit File Count**: Graph view works best with < 10,000 files
- **Reduce Connections**: Too many WikiLinks can slow down rendering
- **Use Filters**: Enable filters to show only relevant files
- **Collapse Groups**: Use `collapse-color-groups: true`

## 📚 **Related Documentation**

- [[docs/canonical/attachments/Obsidian-WikiLink-Format-Guide|WikiLink Format Guide]] - Proper WikiLink usage
- [[docs/canonical/MCP-Self-Improvement-Guide|MCP Self-Improvement Guide]] - How MCP tools improve documentation
- [[docs/canonical/Documentation-Status-Report|Documentation Status Report]] - Overall documentation status

## 🎯 **Quick Fix Checklist**

- [ ] **Restart Obsidian** completely
- [ ] **Verify vault path** is correct
- [ ] **Check file permissions** for Obsidian access
- [ ] **Clear graph cache** if it exists
- [ ] **Refresh graph view** with Ctrl+R
- [ ] **Verify configuration** files are valid JSON
- [ ] **Check for broken WikiLinks** and fix them
- [ ] **Ensure markdown files** exist in canonical directories

---

*This guide provides comprehensive solutions for Obsidian graph view issues based on research and best practices.* 