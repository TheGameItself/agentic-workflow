# EARS Requirement Template

```mmcp
<!-- MMCP-START -->
λ:ears_wrapper(
  [ ] #".root"# {protocol:"MCP", version:"1.3.0", standard:"PFSUS+EARS+LambdaJSON"}
  
  ## {type:EARS, 
      id:"EARS-XXX", 
      entity:"[System/Component Name]", 
      action:"shall", 
      response:"[Required behavior/response]", 
      condition:"[Optional: when/if condition]",
      priority:"medium",
      verification:"[Verification method]",
      rationale:"[Why this requirement exists]",
      source:"[Requirement source]",
      tags:["tag1", "tag2", "tag3"],
      risk_level:"medium",
      compliance:["standard1", "standard2"]}
  
  ## Requirement Statement
  
  **[Entity]** **[Action]** **[Response]** **[Condition]**.
  
  ### Details
  
  - **ID**: EARS-XXX
  - **Priority**: [low/medium/high/critical]
  - **Verification**: [unit_test/integration_test/manual_test/inspection/analysis]
  - **Risk Level**: [low/medium/high]
  - **Source**: [Stakeholder/Document]
  
  ### Rationale
  
  [Explanation of why this requirement is necessary]
  
  ### Acceptance Criteria
  
  ```gherkin
  Given [initial condition]
  When [action occurs]
  Then [expected result]
  ```
  
  ### Dependencies
  
  - @{EARS-XXX} [Description of dependency]
  - @{EARS-XXX} [Description of dependency]
  
  ### Compliance
  
  - [Standard/Regulation 1]
  - [Standard/Regulation 2]
  
  ### Verification Plan
  
  [Detailed plan for how this requirement will be verified]
  
  ### Notes
  
  [Additional notes, constraints, or considerations]
)
<!-- MMCP-END -->
```

## Usage Instructions

1. Replace `EARS-XXX` with the next sequential EARS ID
2. Fill in the entity (system, component, user, etc.)
3. Choose appropriate action verb (shall, should, may, will, must)
4. Describe the required response or behavior
5. Add condition if the requirement is conditional
6. Set priority and risk level appropriately
7. Define verification method
8. Add rationale explaining why the requirement exists
9. Include acceptance criteria in Gherkin format
10. List any dependencies on other requirements
11. Note relevant compliance standards
12. Describe the verification plan

## EARS Shorthand Notation

For quick entry, use the shorthand format:

```
E:[Entity] A:[Action] R:[Response] C:[Condition] #EARS-XXX ![Priority] @[Verification]
```

Example:
```
E:System A:shall R:authenticate_user C:login_attempt #EARS-001 !high @unit_test
```

## Common Action Verbs

- **shall**: Mandatory requirement (strongest)
- **should**: Recommended requirement
- **may**: Optional requirement
- **will**: Statement of fact or intention
- **must**: Absolute requirement (use sparingly)

## Verification Methods

- **unit_test**: Automated unit testing
- **integration_test**: Automated integration testing
- **system_test**: Full system testing
- **manual_test**: Manual testing procedures
- **inspection**: Code or design review
- **analysis**: Mathematical or logical analysis
- **demonstration**: Live demonstration
- **simulation**: Simulation or modeling

## Priority Levels

- **critical**: System cannot function without this
- **high**: Important for system operation
- **medium**: Standard requirement
- **low**: Nice to have feature

## Risk Levels

- **high**: Significant impact if not implemented
- **medium**: Moderate impact if not implemented
- **low**: Minimal impact if not implemented