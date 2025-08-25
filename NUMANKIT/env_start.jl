using Plots
using SparseArrays
using LinearAlgebra
using JLD
using Arpack
using MatrixEquations
using IterativeSolvers
using Distributed
using GenericSchur
using IncompleteLU
using JacobiDavidson
include("basic_tools.jl")
include("discretization.jl")
include("continuation.jl")
include("time_integration.jl")
include("codim1_bifurcation_analysis.jl")
include("minimally_augmented_systems.jl")
include("eigsolvers.jl")