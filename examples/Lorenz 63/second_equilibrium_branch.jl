# the central analyse_branch() function conveniently gives us an approximate tangent at the branchpoint
# with this and the actual branchpoint, we can initialize a new branch using the switch_branch() function.
# To do this we feed the function the jacobian, the branchpoint and the approximate tangent vector to this point on the previous branch
# and we get in return a new tangent vector (this time along the new branch)
Vnew=switch_branch(bifurcation_points.BP.BP, bifurcation_points.BP.tangents, DF);

# we then resume continuation by setting
X1=bifurcation_points.BP.BP;
v1=Vnew;
# and choosing a number of steps in the first (left) direction:
steps=10;
Branch2=cont(initial_stepsize,maximum_stepsize,maximum_number_of_newton_iterations,steps,order_of_newton_method,F,DF,X1,v1,direction);
# we now go in the opposite direction, to that aim we first reverse the new branch:
Branch2=reverse(Branch2,dims=2);
# we then readjust the number of steps and change the direction
steps=20
direction=-1;
# and finally we add the part of the branch we're about to compute to the pre-existing new branch:
Branch2=[Branch2 cont(initial_stepsize,maximum_stepsize,maximum_number_of_newton_iterations,steps,order_of_newton_method,F,DF,X1,v1,direction);]

continuation_parameter_values=Branch2[end,:];
variable_of_interest=Branch2[1,:];
plot!(continuation_parameter_values,variable_of_interest,label="")

# the reader is left free to check for themselves that the new branch contains no new local bifurcations (you can take inspiration from what was done in the "first_equilibrium_branch.jl" file)...
# Finally, let's see what's up with those Hopf bifurcations! Are they super or subcritical? (see the file "phase_exploration_hopf.jl")