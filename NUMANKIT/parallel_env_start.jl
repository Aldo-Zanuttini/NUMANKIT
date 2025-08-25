n=3; # number of processors to add
addprocs(n)
@everywhere include("C:\\............\\numerics kit\\env_start.jl") # absolute path to basic_tools.jl
@everywhere include("C:\\............\\my_functions.jl") # absolute path to the functions you defined for this project

# remove all added processors: rmprocs(workers())