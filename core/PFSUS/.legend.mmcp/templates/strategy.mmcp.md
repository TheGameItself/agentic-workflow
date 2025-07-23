# STRATEGY Guide Template

```mmcp
<!-- MMCP-START -->
Δ:strategy_wrapper(
  [ ] #".root"# {protocol:"MCP", version:"1.3.0", standard:"PFSUS+EARS+STRATEGY"}
  
  ## {type:STRATEGY, 
      id:"STRAT-[CATEGORY]-XXX", 
      category:"[code_practices/architecture/optimization/security/testing/deployment]",
      title:"[Strategy Title]",
      version:"1.0.0",
      author:"[Author Name]",
      last_modified:"2025-07-21T12:00:00Z",
      tags:["strategy", "best_practices", "guidelines"],
      scope:"[project/team/organization]",
      maturity:"[draft/review/approved/deprecated]"}
  
  # [Strategy Title]
  
  ## Overview
  
  [Brief description of what this strategy covers and why it's important]
  
  ## Principles
  
  ### Core Principles
  - **[Principle 1]**: [Description and rationale]
  - **[Principle 2]**: [Description and rationale]
  - **[Principle 3]**: [Description and rationale]
  
  ### Supporting Principles
  - [Additional principle with brief description]
  - [Additional principle with brief description]
  
  ## Best Practices
  
  ### [Category 1]
  
  #### Do's ✅
  - [Recommended practice with explanation]
  - [Recommended practice with explanation]
  
  #### Don'ts ❌
  - [Practice to avoid with explanation]
  - [Practice to avoid with explanation]
  
  #### Examples
  
  ```[language]
  // Good example
  [code example showing best practice]
  ```
  
  ```[language]
  // Bad example (avoid)
  [code example showing what not to do]
  ```
  
  ### [Category 2]
  
  [Repeat structure as above]
  
  ## Implementation Guidelines
  
  ### Getting Started
  1. [First step with details]
  2. [Second step with details]
  3. [Third step with details]
  
  ### Gradual Adoption
  - **Phase 1**: [Initial implementation steps]
  - **Phase 2**: [Intermediate implementation steps]
  - **Phase 3**: [Advanced implementation steps]
  
  ### Tools and Resources
  - **[Tool Name]**: [Description and use case]
  - **[Resource Name]**: [Description and link if applicable]
  
  ## Metrics and Measurement
  
  ### Key Performance Indicators (KPIs)
  - **[Metric Name]**: [Description, target value, measurement method]
  - **[Metric Name]**: [Description, target value, measurement method]
  
  ### Success Criteria
  - [Criterion 1 with measurable outcome]
  - [Criterion 2 with measurable outcome]
  
  ## Common Pitfalls
  
  ### [Pitfall Category]
  - **Problem**: [Description of the pitfall]
  - **Impact**: [What happens when this occurs]
  - **Solution**: [How to avoid or resolve]
  - **Prevention**: [Proactive measures]
  
  ## Decision Framework
  
  ### When to Apply This Strategy
  - [Condition 1]
  - [Condition 2]
  - [Condition 3]
  
  ### When NOT to Apply This Strategy
  - [Exception 1]
  - [Exception 2]
  - [Exception 3]
  
  ## Related Strategies
  
  - @{STRAT-XXX-XXX} [Related strategy with brief description]
  - @{STRAT-XXX-XXX} [Related strategy with brief description]
  
  ## References and Further Reading
  
  - [Book/Article Title] by [Author]
  - [Online Resource] - [URL]
  - [Standard/Framework] - [Reference]
  
  ## Changelog
  
  - **v1.0.0** (2025-07-21): Initial version
  - **v0.9.0** (2025-07-20): Draft version for review
)
<!-- MMCP-END -->
```

## Strategy Categories

### Code Practices (STRAT-CP-XXX)
- Coding standards and conventions
- Code quality and maintainability
- Documentation practices
- Code review processes

### Architecture (STRAT-ARCH-XXX)
- System design principles
- Architectural patterns
- Component organization
- Scalability considerations

### Optimization (STRAT-OPT-XXX)
- Performance optimization
- Resource utilization
- Caching strategies
- Database optimization

### Security (STRAT-SEC-XXX)
- Security best practices
- Authentication and authorization
- Data protection
- Vulnerability management

### Testing (STRAT-TEST-XXX)
- Testing strategies
- Test automation
- Quality assurance
- Test coverage

### Deployment (STRAT-DEPLOY-XXX)
- Deployment strategies
- CI/CD practices
- Environment management
- Release management

### Operations (STRAT-OPS-XXX)
- Monitoring and alerting
- Incident response
- Capacity planning
- Maintenance procedures

## Usage Guidelines

1. **Choose Appropriate Category**: Select the category that best fits your strategy
2. **Use Sequential IDs**: Increment the XXX number for each new strategy in a category
3. **Be Specific**: Focus on actionable, specific guidance rather than general advice
4. **Include Examples**: Provide concrete examples of do's and don'ts
5. **Measure Success**: Define clear metrics for evaluating strategy effectiveness
6. **Keep Updated**: Regularly review and update strategies as practices evolve
7. **Cross-Reference**: Link to related strategies and external resources

## Template Customization

This template can be customized for specific domains:

- **Frontend Strategy**: Focus on UI/UX, component libraries, state management
- **Backend Strategy**: Focus on APIs, data processing, service architecture
- **DevOps Strategy**: Focus on infrastructure, automation, monitoring
- **Data Strategy**: Focus on data modeling, ETL, analytics
- **Mobile Strategy**: Focus on mobile-specific considerations

## Shorthand Notation

For quick strategy references:

```
S:[Category] T:[Title] P:[Principle] BP:[Best Practice] #STRAT-[CAT]-XXX
```

Example:
```
S:CodePractices T:CleanCode P:Readability BP:DescriptiveNames #STRAT-CP-001
```