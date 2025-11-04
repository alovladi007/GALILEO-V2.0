# Examples

This directory contains example scripts demonstrating the use of the GeoSense Platform.

## Available Examples

### session1_demo.py

Comprehensive demonstration of Session 1 physics implementations including:
- Orbital dynamics (Keplerian, perturbations)
- Formation flying (Hill-Clohessy-Wiltshire equations)
- Gravity field modeling
- Numerical propagation

**Usage:**
```bash
python examples/session1_demo.py
```

## Running Examples

Make sure the package is installed first:

```bash
# Install in development mode
pip install -e .

# Run an example
python examples/session1_demo.py
```

## Creating New Examples

When adding new examples:
1. Use clear, descriptive filenames
2. Include docstrings explaining the purpose
3. Add comments for key steps
4. Keep examples focused on specific features
5. Update this README with the new example

## Example Output

Examples typically generate:
- Plots showing results
- Printed statistics and metrics
- Validation of physics models
- Performance benchmarks
