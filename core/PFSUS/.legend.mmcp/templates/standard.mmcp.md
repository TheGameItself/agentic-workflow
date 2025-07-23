```mmcp
<!-- MMCP-START -->
%% Copyright 2025 Kalxi. All rights reserved. See header for terms. %%
[ ] #".root"# {protocol:"MCP", version:"1.2.1", standard:"PFSUS+EARS+LambdaJSON"}
# {TITLE} v{VERSION}
## {type:Meta, author:"{AUTHOR}", license:"MIT", last_modified:"{DATE}T00:00:00Z", id:"{ID}"}
## {type:Schema, $schema:"https://json-schema.org/draft/2020-12/schema", required:["type","id","version"], properties:{type:{type:"string"},id:{type:"string"},version:{type:"string"},last_modified:{type:"string",format:"date-time"},author:{type:"string"}}}
## {type:Changelog, entries:[{"{DATE}":"Initial version."}]}

## {type:{TYPE}, id:"{ID}", desc:"{DESCRIPTION}"}

## 1. Introduction

{INTRODUCTION}

## 2. {SECTION_TITLE}

{SECTION_CONTENT}

## {type:SelfReference, file:"{FILENAME}", version:"{VERSION}", checksum:"sha256:{CHECKSUM}", canonical_address:"{ADDRESS}", self_repair:{desc:"If checksum fails, fetch canonical version from CoreManifest.", source:"CoreManifest"}, project_agnostic:true}

@{visual-meta-start}
author = {{AUTHOR}},
title = {{TITLE}},
version = {{VERSION}},
structure = { introduction, {STRUCTURE} },
@{visual-meta-end}
%% MMCP-FOOTER: version={VERSION}; timestamp={DATE}T00:00:00Z; checksum=sha256:{CHECKSUM}; author={AUTHOR}; license=MIT
<!-- MMCP-END -->
```