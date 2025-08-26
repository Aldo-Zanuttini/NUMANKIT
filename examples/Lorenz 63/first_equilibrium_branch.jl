# begin a continuation problem: get a starting point on an equilibrium
# manifold and the associated tangent vector
initial_parameter_value=30; # specify an initial parameter to start the continuation from
initial_equilibrium_guess=[sqrt(beta*(initial_parameter_value-1));sqrt(beta*(initial_parameter_value-1));initial_parameter_value-1] # here you can also guess a random equilibrium, but to ensure the flow of this example is "as expected" we input an exact equilibrium
X0,v0=init_cont(initial_parameter_value,F,DF,initial_equilibrium_guess); # get the first point on the bifurcation diagram and its associated tangent vector
v0=sign(v0[end])*v0/norm(v0); # normalize and ensure that the vector points rightwards in the continuation parameter direction

# specify the parameters for continuation:
initial_stepsize=0.1;
maximum_stepsize=2;
maximum_number_of_newton_iterations=9; # maximum number of newton iterations before the stepsize is halved
steps=49; # number of points we want to compute on the bifurcation diagram
order_of_newton_method=5; # order of the moore-penrose corrections (takes values from 2 to 5)
direction=1; # direction to perform the continuation in (leftwards in this case)
# we should also define F(x) and DF(x), however we already did this in the file "my_system"

# obtain a Branch
Branch=cont(initial_stepsize,maximum_stepsize,maximum_number_of_newton_iterations,steps,order_of_newton_method,F,DF,X0,v0,direction);

# produce a plot of the branch,
continuation_parameter_values=Branch[end,:];
variable_of_interest=Branch[1,:]; # for this particular system the second variable is useless
plot(continuation_parameter_values,variable_of_interest,label="")
xlabel!("α")
ylabel!("x")


# analyse the branch:
# define the parameters for the analyse_branch function
tolerance=1e-6; # accuracy of the bifurcation points detected
maximum_number_of_iterations=500; # maximum number of iterations (for all methods used)
k=2; # size of the projected problem in the Meerbergen Spence algorithm (see: https://doi.org/10.1137/110827600)
hopf=true; # look for Hopf bifurcations
branchpoint=true; # look for branchpoints
fold=true; # look for fold points
bifurcation_points=analyse_branch(Branch,F,DF,tol=tolerance,maxiter=maximum_number_of_iterations,k=k,hopf=hopf,fold=fold,branchpoint=branchpoint)

#### NOTE: the above can be done in parallel for hopf, branchpoint and fold. This could save you a lot of time potentially. To do this, run the "parallel_env_start.jl" file with the necessary adjustments (specify the absolute path to your "NUMANKIT" folder), please remember to reset the number of working processors to 1 when you're done doing the analysis!!! This can be done via the "rmprocs(workers())" command.

# normally here you should check that bifurcation_points contains hopf, fold and/or branchpoints. In this case we know the Lorenz system, so we know we'll find all three types of singularity
hopf_points=bifurcation_points.H.H; # extract the fold
scatter!([hopf_points[end,:]],[hopf_points[2,:]],ms=4,label="H")
fold_points=bifurcation_points.LP.LP # extract the hopfs
scatter!([fold_points[end,:]],[fold_points[2,:]],ms=4,label="LP")
branch_points=bifurcation_points.BP.BP # extract the branchpoints
scatter!([branch_points[end,:]],[branch_points[2,:]],ms=4,label="BP")
display(plot!(legend=true))


# Okay: we have a branchpoint. Let's compute the other branch too! (see "second_equilibrium_branch.jl" file)