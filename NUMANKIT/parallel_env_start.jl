n=3; # number of processors to add
addprocs(n)
@everywhere using Plots
@everywhere using SparseArrays
@everywhere using LinearAlgebra
@everywhere using JLD
@everywhere using Arpack
@everywhere using MatrixEquations
@everywhere using IterativeSolvers
@everywhere using Distributed
@everywhere include("C:\\............\\numerics kit\\basic_tools.jl") # absolute path to basic_tools.jl
@everywhere include("C:\\............\\numerics kit\\discretization.jl") # absolute path to discretization.jl
@everywhere include("C:\\............\\numerics kit\\continuation.jl") # absolute path to continuation.jl
@everywhere include("C:\\............\\numerics kit\\time_integration.jl") # absolute path to time_integration.jl
@everywhere include("C:\\............\\numerics kit\\codim1_bifurcation_analysis.jl") # absolute path to codim1_bifurcation_analysis
@everywhere include("C:\\............\\my_functions.jl") # absolute path to the functions you defined for this project

# remove all added processors: rmprocs(workers())