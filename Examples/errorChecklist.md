# Error checklist for examples and strategies
<p>
Strategy, algorithm, result, iterations, time, log(loss) about
<p>
  
## Level-set 3D (ADAM 0.001)  {3>16>1}
- QuadratureTraining CubaCuhre **[OK]** 2000 / 553 s / 10^-5
- QuadratureTraining HCubatureJL **[OK]** 2000 / 456 s / 10^-5.3
- QuadratureTraining CubatureJLh **[OK]** 2000 / 462 s / 10^-5.3
- QuadratureTraining CubatureJLp **[OK]** 2000 / 355 s / 10^-5
- StochasticTraining **[OK]** 2000 / 362 s / 10^-1.7
- GridTraining **[Extremely slow]** 
- QuasiRandomTraining Uniform Sampling **[OK]** 2000 / 338 s / 10^-3.1

## Allen-Cahn 5D (ADAM 0.02)  {5>20>1}
- QuadratureTraining CubaCuhre **[OK]** 1000 / 596 s / 8.5*10^-9
- QuadratureTraining HCubatureJL **[OK]** 1000 / 262 s / 8.5 * 10^-9
- QuadratureTraining CubatureJLh **[OK]** 1000 / 240 s / 8.0 * 10^-9 
- QuadratureTraining CubatureJLp **[OK]** 1000 / 852 s / 1.6 * 10^-8
- StochasticTraining **Cannot run** `InexactError Int64`
- GridTraining **[Extremely slow]** 
- QuasiRandomTraining Uniform Sampling **[OK]** 1000 / 265 s / 6.3 * 10^-5

## Hamilton-Jacobi 5D (ADAM 0.004)  {5>20>1}
- QuadratureTraining CubaCuhre **[OK]** 500 / 453 s / 3.9 * 10^-6
- QuadratureTraining HCubatureJL **[OK]** 500 / 218 s / 4.3 * 10^-7
- QuadratureTraining CubatureJLh **[OK]** 500 / 222 s / 4.3 * 10^-7
- QuadratureTraining CubatureJLp **[OK]** 500 / 687 s / 7.0 * 10^-7
- StochasticTraining **Cannot run** `InexactError Int64`
- GridTraining **[Extremely slow]** 
- QuasiRandomTraining Uniform Sampling **[OK]** 500 / 170 s / 5.3 * 10^-3

## Nernst-Plank
- To do
