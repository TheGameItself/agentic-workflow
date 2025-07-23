# Mathematical Expression Template

```mmcp
<!-- MMCP-START -->
Δ:mathematical_wrapper(
  [ ] #".root"# {protocol:"MCP", version:"1.4.0", standard:"PFSUS+EARS+LambdaJSON+MathOps"}
  
  ## {type:Mathematical, 
      id:"MATH-XXX", 
      expression:"[Mathematical expression in standard notation]",
      latex:"[LaTeX representation]",
      operators:[
        {
          token:"[OPERATOR_TOKEN]",
          symbol:"[Unicode symbol]",
          unicode:"U+XXXX",
          latex:"[LaTeX command]",
          precedence:[1-10],
          arity:[0-3]
        }
      ],
      wrapper_assignment:{
        primary:"[λ|ℵ|Δ|β|Ω|i|τ]",
        secondary:["[additional wrappers]"],
        auto_detected:true
      },
      complexity:{
        time:{best:"O(1)", average:"O(n)", worst:"O(n²)"},
        space:"O(1)",
        notation:"BIGO"
      },
      domain:{
        input_set:"REALS",
        output_set:"REALS",
        constraints:["domain restrictions"]
      },
      properties:{
        commutative:true,
        associative:true,
        distributive:false,
        identity_element:"0",
        monotonic:true
      }}
  
  # [Mathematical Expression Title]
  
  ## Expression
  
  **Standard Notation**: [expression]
  
  **LaTeX**: `[latex_code]`
  
  **Rendered**: [rendered_expression]
  
  ## Operators Used
  
  ### Primary Operators
  - **[OPERATOR_NAME]** ([symbol]): [description]
    - Unicode: U+XXXX
    - LaTeX: `\command`
    - Precedence: [1-10]
    - Arity: [number of operands]
  
  ### Secondary Operators
  - **[OPERATOR_NAME]** ([symbol]): [description]
  
  ## Wrapper Assignment
  
  **Primary Wrapper**: [wrapper] ([reason for assignment])
  
  **Precedence Chain**: [full wrapper chain if nested]
  
  **Auto-Detection**: [Yes/No] - [explanation if manual]
  
  ## Mathematical Properties
  
  ### Algebraic Properties
  - **Commutativity**: [Yes/No] - [explanation]
  - **Associativity**: [Yes/No] - [explanation]  
  - **Distributivity**: [Yes/No] - [over which operations]
  - **Identity Element**: [element] - [verification]
  - **Inverse Element**: [element] - [conditions]
  
  ### Analytical Properties
  - **Continuity**: [continuous/discontinuous] - [at which points]
  - **Differentiability**: [differentiable/non-differentiable] - [conditions]
  - **Monotonicity**: [monotonic/non-monotonic] - [intervals]
  - **Boundedness**: [bounded/unbounded] - [bounds if applicable]
  
  ## Domain and Codomain
  
  **Input Domain**: [set] ([description])
  
  **Output Codomain**: [set] ([description])
  
  **Constraints**: 
  - [constraint 1]
  - [constraint 2]
  
  **Special Cases**:
  - [special case 1]: [behavior]
  - [special case 2]: [behavior]
  
  ## Complexity Analysis
  
  ### Time Complexity
  - **Best Case**: [complexity] - [conditions]
  - **Average Case**: [complexity] - [typical conditions]
  - **Worst Case**: [complexity] - [worst conditions]
  
  ### Space Complexity
  - **Space**: [complexity] - [memory requirements]
  
  ### Asymptotic Notation
  - **Big-O**: O([bound]) - [upper bound analysis]
  - **Big-Ω**: Ω([bound]) - [lower bound analysis]  
  - **Big-Θ**: Θ([bound]) - [tight bound analysis]
  
  ## Proof (if applicable)
  
  ### Theorem Statement
  **Theorem**: [formal statement]
  
  ### Assumptions
  - [assumption 1]
  - [assumption 2]
  
  ### Proof Steps
  1. **Step 1**: [statement]
     - **Justification**: [reasoning]
     - **Operators**: [operators used]
  
  2. **Step 2**: [statement]
     - **Justification**: [reasoning]
     - **Operators**: [operators used]
  
  ### Conclusion
  **Therefore**: [conclusion] ∎
  
  **Proof Method**: [direct/contradiction/induction/construction]
  
  ## Numerical Methods
  
  ### Approximation Methods
  - **Method**: [numerical method]
  - **Error Bound**: [error analysis]
  - **Convergence Rate**: [rate of convergence]
  
  ### Stability Analysis
  - **Numerical Stability**: [stable/conditionally stable/unstable]
  - **Condition Number**: [if applicable]
  
  ### Implementation Notes
  - [implementation consideration 1]
  - [implementation consideration 2]
  
  ## Visualization
  
  ### Graph Type
  **Type**: [function/relation/set/proof_tree/complexity_chart]
  
  ### Dimensions
  **Space**: [1D/2D/3D/4D] visualization
  
  ### Interactive Elements
  - **Parameters**: [adjustable parameters]
  - **Animation**: [animated elements]
  - **Zoom/Pan**: [navigation capabilities]
  
  ### Visualization Code
  ```python
  # Example visualization code
  import matplotlib.pyplot as plt
  import numpy as np
  
  # [visualization implementation]
  ```
  
  ## Applications
  
  ### [Domain 1] (e.g., Physics)
  **Description**: [how the expression is used]
  
  **Examples**:
  - [specific example 1]
  - [specific example 2]
  
  ### [Domain 2] (e.g., Engineering)
  **Description**: [how the expression is used]
  
  **Examples**:
  - [specific example 1]
  - [specific example 2]
  
  ## Related Expressions
  
  ### Generalizations
  - @{MATH-XXX} [More general form]
  - @{MATH-XXX} [Extended version]
  
  ### Special Cases
  - @{MATH-XXX} [Simplified form]
  - @{MATH-XXX} [Restricted domain]
  
  ### Inverse Operations
  - @{MATH-XXX} [Inverse expression]
  - @{MATH-XXX} [Dual expression]
  
  ## References
  
  ### Theorems
  - **[Theorem Name]**: [source] - @{THEOREM-XXX}
  - **[Lemma Name]**: [source] - @{LEMMA-XXX}
  
  ### Definitions
  - **[Definition Name]**: [source] - @{DEF-XXX}
  
  ### Historical Notes
  - **Discovered by**: [mathematician] ([year])
  - **Historical Context**: [background information]
  
  ## Implementation Examples
  
  ### Python Implementation
  ```python
  def mathematical_expression(x, *args):
      """
      Implementation of [expression name].
      
      Args:
          x: Input value(s)
          *args: Additional parameters
          
      Returns:
          Result of the mathematical expression
          
      Complexity: O([complexity])
      """
      # Implementation here
      return result
  ```
  
  ### Mathematical Software
  ```mathematica
  (* Mathematica implementation *)
  expression[x_] := [mathematica_code]
  ```
  
  ```matlab
  % MATLAB implementation
  function result = expression(x)
      % [matlab_code]
  end
  ```
  
  ## Testing and Validation
  
  ### Test Cases
  ```python
  # Unit tests for the mathematical expression
  def test_mathematical_expression():
      # Test case 1: [description]
      assert expression(input1) == expected1
      
      # Test case 2: [description]  
      assert expression(input2) == expected2
      
      # Edge case: [description]
      assert expression(edge_input) == edge_expected
  ```
  
  ### Validation Methods
  - **Analytical Verification**: [method]
  - **Numerical Verification**: [method]
  - **Graphical Verification**: [method]
  
  ## Extensions and Variations
  
  ### Multivariable Extension
  **Expression**: [multivariable form]
  **Additional Operators**: [new operators needed]
  
  ### Complex Number Extension
  **Expression**: [complex form]
  **Domain**: ℂ (complex numbers)
  
  ### Vector/Matrix Extension
  **Expression**: [vector/matrix form]
  **Operators**: [vector/matrix operators]
  
  ## Troubleshooting
  
  ### Common Issues
  - **Issue 1**: [problem description]
    - **Cause**: [root cause]
    - **Solution**: [resolution]
  
  - **Issue 2**: [problem description]
    - **Cause**: [root cause]
    - **Solution**: [resolution]
  
  ### Numerical Pitfalls
  - **Overflow/Underflow**: [conditions and prevention]
  - **Loss of Precision**: [when it occurs and mitigation]
  - **Convergence Issues**: [convergence problems and solutions]
)
<!-- MMCP-END -->
```

## Mathematical Operator Quick Reference

### Arithmetic Operators
- **⊕** (ADD): Addition or direct sum
- **⊖** (SUB): Subtraction or symmetric difference  
- **⊗** (MUL): Multiplication or tensor product
- **⊘** (DIV): Division or quotient
- **⋅** (DOT): Dot product or scalar multiplication
- **％** (PERCNT): Percentage or modulo
- **↑** (POW): Exponentiation or power
- **!** (FACT): Factorial operation

### Comparison Operators
- **≡** (EQL): Equivalence relation
- **≢** (NEQL): Non-equivalence
- **≺** (LESS): Strict ordering (less than)
- **≻** (GREAT): Strict ordering (greater than)
- **⪯** (LEQ): Weak ordering (less or equal)
- **⪰** (GEQ): Weak ordering (greater or equal)
- **≈** (APPRX): Approximate equality
- **≅** (IDEN): Congruence or isomorphism

### Set Theory Operators
- **∈** (ELM): Element membership
- **∉** (NELM): Non-membership
- **∅** (EMSET): Empty set
- **∩** (INTSC): Set intersection
- **∪** (UNION): Set union
- **⊂** (PSUB): Proper subset
- **⊆** (SBEQ): Subset or equal
- **⊄** (NSUB): Not a subset
- **△** (SDIFF): Symmetric difference

### Logical Operators
- **∀** (FALL): Universal quantifier
- **∃** (EXIST): Existential quantifier
- **∧** (LAND): Logical conjunction
- **∨** (LOR): Logical disjunction
- **¬** (LNOT): Logical negation
- **⊕** (XOR): Exclusive or
- **⊼** (NAND): Not and (Sheffer stroke)
- **⊽** (NOR): Not or (Peirce arrow)
- **→** (IMPL): Logical implication
- **↔** (IFF): Logical equivalence

### Calculus Operators
- **∑** (SUM): Summation operator
- **∏** (PROD): Product operator
- **∇** (GRAD): Gradient operator
- **∂** (PARD): Partial derivative
- **d/dx** (DERIV): Total derivative
- **′** (PRIME): Prime notation for derivatives
- **∫** (INTRG): Integral operator
- **∬** (DINT): Double integral
- **∭** (TINT): Triple integral

### Function Operators
- **∘** (COMP): Function composition
- **↦** (MTO): Maps to (function mapping)
- **√** (SQRT): Square root
- **∛** (CBRT): Cube root
- **∜** (QRT): Fourth root

## Usage Guidelines

1. **Choose Appropriate Operators**: Select operators that match the mathematical context
2. **Use Correct Precedence**: Follow mathematical precedence rules
3. **Specify Wrapper Assignment**: Choose the appropriate calculus notation wrapper
4. **Include Complexity Analysis**: Provide computational complexity when relevant
5. **Document Properties**: Specify algebraic and analytical properties
6. **Provide Examples**: Include concrete examples and applications
7. **Test Thoroughly**: Validate mathematical expressions with test cases
8. **Consider Numerical Issues**: Address potential numerical computation problems

## Template Customization

This template can be customized for specific mathematical domains:

- **Pure Mathematics**: Focus on proofs, theorems, and abstract properties
- **Applied Mathematics**: Emphasize applications and numerical methods
- **Computer Science**: Include algorithmic complexity and implementation
- **Engineering**: Focus on practical applications and approximations
- **Physics**: Include physical interpretations and units