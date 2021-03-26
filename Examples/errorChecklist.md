# Error checklist for examples and strategies 
## Allen-Cahn
- QuadratureTraining CubaCuhre *[OK]*
- QuadratureTraining HCubatureJL *[OK]*
- QuadratureTraining CubatureJLh *[OK]*
- QuadratureTraining CubatureJLp *[OK]*
- StochasticTraining *Cannot run* `InexactError Int64`
- GridTraining *[Extremely slow]*
- QuasiRandomTraining Uniform Sampling *Problems* in `UniformSample()` method definition
