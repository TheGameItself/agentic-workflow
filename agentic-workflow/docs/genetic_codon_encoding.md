# Genetic Codon Encoding

The MCP genetic trigger system uses a 256-codon (4-letter) DNA-inspired encoding for all genetic data. This ensures bijective, lossless mapping between bytes and codons, supporting robust storage and transmission.

## Rationale
- 256 unique codons (A/T/G/C quadruplets) map directly to byte values (0-255)
- Enables encoding of arbitrary binary data as DNA-like strings
- Supports advanced genetic operations and cross-system compatibility

## Implementation
- Encoding: each byte maps to a unique 4-letter codon
- Decoding: each 4-letter codon maps back to its byte
- Used for dna_signature, epigenetic markers, and hormone feedback

## Usage
- See `GeneticTrigger.encode_dict_to_codon` and `decode_codon_to_dict`
- Used throughout the genetic trigger and epigenetic memory systems

## Related Docs
- [[genetic_trigger_system]]
- [[split_brain_testing]]
- [[epigenetic_memory]] 