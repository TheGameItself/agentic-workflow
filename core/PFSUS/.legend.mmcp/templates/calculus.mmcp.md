```mmcp
<!-- MMCP-START -->
%% Copyright 2025 Kalxi. All rights reserved. See header for terms. %%
[ ] #".root"# {protocol:"MCP", version:"1.2.1", standard:"PFSUS+EARS+LambdaJSON"}
# {TITLE} v{VERSION}
## {type:Meta, author:"{AUTHOR}", license:"MIT", last_modified:"{DATE}T00:00:00Z", id:"{ID}"}
## {type:Schema, $schema:"https://json-schema.org/draft/2020-12/schema", required:["type","id","version","expressions"], properties:{type:{type:"string",enum:["Calculus","LambdaCalculus","QuantumCalculus","CategoryCalculus"]},id:{type:"string"},version:{type:"string"},expressions:{type:"array",items:{type:"object",required:["id","expression"],properties:{id:{type:"string"},expression:{type:"string"},nested:{type:"array"}}}}}}
## {type:Changelog, entries:[{"{DATE}":"Initial version."}]}

## {type:{CALCULUS_TYPE}, id:"{ID}", desc:"{DESCRIPTION}"}

## 1. Introduction

{INTRODUCTION}

## 2. Expressions

### 2.1 Basic Expressions

λ:expression({
  id: "expr-001",
  expression: "λx.x",
  description: "Identity function"
})

λ:expression({
  id: "expr-002",
  expression: "λx.λy.x",
  description: "Constant function (K combinator)"
})

### 2.2 Nested Expressions

λ:expression({
  id: "nested-001",
  expression: "λf.λg.λx.(f (g x))",
  description: "Function composition",
  nested: [
    {
      id: "inner-001",
      expression: "g x",
      description: "Apply g to x"
    },
    {
      id: "inner-002",
      expression: "f (g x)",
      description: "Apply f to the result of (g x)"
    }
  ]
})

## 3. Composition Rules

1. **Order Agnostic Nesting**: Nested expressions can be composed in any order, with the final result determined by the explicit dependencies rather than the nesting order.
2. **Dependency Resolution**: When expressions have dependencies, they are resolved before composition.
3. **Recursive Composition**: Expressions can be recursively composed to arbitrary depth.

## 4. Examples

### 4.1 Order Agnostic Composition

```
λ:compose([
  λ:expression({ id: "expr-001", expression: "λx.x" }),
  λ:expression({ id: "expr-002", expression: "λx.λy.x" })
])
```

This composition is equivalent to:

```
λ:compose([
  λ:expression({ id: "expr-002", expression: "λx.λy.x" }),
  λ:expression({ id: "expr-001", expression: "λx.x" })
])
```

Because the composition is order agnostic, with the result determined by the dependency graph rather than the order of specification.

### 4.2 Nested Composition with Dependencies

```
λ:compose({
  expression: "λf.λg.λx.(f (g x))",
  dependencies: [
    { id: "dep-001", expression: "g x" },
    { id: "dep-002", expression: "f y", dependencies: [{ id: "dep-001", as: "y" }] }
  ]
})
```

## {type:SelfReference, file:"{FILENAME}", version:"{VERSION}", checksum:"sha256:{CHECKSUM}", canonical_address:"{ADDRESS}", self_repair:{desc:"If checksum fails, fetch canonical version from CoreManifest.", source:"CoreManifest"}, project_agnostic:true}

@{visual-meta-start}
author = {{AUTHOR}},
title = {{TITLE}},
version = {{VERSION}},
structure = { introduction, expressions, composition_rules, examples },
@{visual-meta-end}
%% MMCP-FOOTER: version={VERSION}; timestamp={DATE}T00:00:00Z; checksum=sha256:{CHECKSUM}; author={AUTHOR}; license=MIT
<!-- MMCP-END -->
```