## [WIP] 1D Poisson-Nernst-Planck equations solution using NeuralPDE

The following model depicts the ion transport of Cl^- and Na^+, solely governed by diffusion and migration, and described by the 1D Poisson–Nernst–Planck equations. The model is a simplification based on [1-3], where an electrochemical treatment was used for tumor ablation. 
The model is solved using NeuralPDE.

###   Mathematical model

####   Equations
        d2Phi/dx2 =    ( 1.0 / Po_1 ) * ( z_Na^+ * Na^+ + z_Cl^- * Cl^- )
        dNa^+/dt =     ( 1.0 / Pe_Na^+ ) * d2Na^+/dx2 
                     + z_Na^+ / ( abs(z_Na^+) * M_Na^+ ) *
                     ( dNa^+/dx * dPhi/dx + Na^+ * d2Phi/dx2 )
        dCl^-/dt =     ( 1.0 / Pe_Cl^- ) * d2Cl^-/dx2
                     + z_Cl^- / ( abs(z_Cl^-) * M_Cl^- ) 
                     * ( dCl^-/dx * dPhi/dx + Cl^- * d2Phi/dx2 )

#### Initial conditions:
        Phi(0,x) = 0.0
        Na^+(0,x) = Na^+_0
        Cl^-(0,x) = Cl^-_0

#### Boundary conditions:

Butler-Volmer equations have been replaced by a linear approximation.

        Phi(t,0) = Phi_0
        Phi(t,n) = 0.0
        Na^+(t,0) = 0.0
        Na^+(t,n) = 2.0 * Na^+_0
        Cl^-(t,0) = 1.37 * Cl^-_0
        Cl^-(t,n) = 0.0
        
#### References 

[1] "pH front tracking in the electrochemical treatment (EChT) of tumors: Experiments and simulations", 
P. Turjanski, N. Olaiz, P. Abou-Adal, C. Suárez 1 , M. Risk 1 , G. Marshall". Electrochimica Acta 54 (2009) 6199–6206.

[2] "Electroterapia y Electroporación en el tratamiento de tumores: modelos teóricos y experimentales". P. Turjanski. Departamento de Computación. Facultad de Ciencias Exactas y Naturales. Universidad de Buenos Aires. 2011.

[3] "Optimal dose-response relationship in electrolytic ablation oftumors with a one-probe-two-electrode device". E. Luján, H. Schinca, N. Olaiz, S. Urquiza, F.V. Molina, P. Turjanski, andG. Marshall. Electrochimica Acta, 186:494–503, December 2015.

        
### Installation and running in GNU/Linux

1) Download Julia from https://julialang.org/downloads/

    E.g.
    ```
        $ wget https://julialang-s3.julialang.org/bin/linux/x64/1.5/julia-1.5.3-linux-x86_64.tar.gz
     ```
2) Extract file
     ```
        $ tar xvzf  julia-1.5.3-linux-x86_64.tar.gz
     ```
3) Copy to /opt and create link
     ```
        $ sudo mv  ./julia-1.5.3 /opt/
        $ sudo ln -s /opt/julia-1.5.3/bin/julia /usr/local/bin/julia
     ```
4) Install required packets
    ```
        $ julia
        julia> import Pkg
        julia> Pkg.add("NeuralPDE")
        julia> Pkg.add("Flux")
        julia> Pkg.add("ModelingToolkit")
        julia> Pkg.add("GalacticOptim")
        julia> Pkg.add("Optim")
        julia> Pkg.add("DiffEqFlux")
        julia> Pkg.add("Plots")
        julia> Pkg.add("Quadrature")
        julia> Pkg.add("Cubature")
        julia> Pkg.add("Cuba")
        julia> Pkg.add("LaTeXStrings")
        julia> Pkg.add("Parameters")
    ```
     
4) Clone project directory
    ```
        $ git clone https://github.com/emmanuellujan/1d-poission-nernst-planck.git
     ```

5) Run

    Solve PNP
    
    ```
        $ julia solve-pnp.jl
        or
        $ julia
        julia> include("1d-poisson-nerst-planck-Cl-Na-adim-neuralpde.jl")
        julia> res, loss, discretization, pars = solve_PNP()
        julia> plot_PNP(res, loss, discretization,pars)
    ```
    
    Convergence test
    
        ```
        $ julia pnp-convergence-test.jl
        ```
