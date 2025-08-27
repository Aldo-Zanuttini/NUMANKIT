# to determine wether or not the cycles resulting from the hopf bifurcations are stable or unstable, one can attempt to look at the first lyapunov coefficient, this is already provided by the analyse_branch() function and can be checked with
show(string("first lyapunov coefficients=",bifurcation_points.H.l1))
# in this case it would seem that the cycles are unstable because both of the first lyapunov coefficients are positive, however this is not always a reliable estimate, so we proceed to check using the eigenvectors
# the analyse_branch() function also returns the eigenvector associated with one of the hopf eigenvalues for each hopf bifurcation discovered. To get this we do:
critical_eigenvector1=bifurcation_points.H.V[:,1];
# a point on the cycle is now of the form X=X*+epsilon*(real(critical_eigenvector1)*cos(theta)+imag(critical_eigenvector1)*sin(theta)) for theta arbitrary, and where X* is the actual equilibrium and epsilon=sqrt(abs(lambda-lambda_c)) is the distance from the true parameter value at which the bifurcation happens. Let us then define
v=real(critical_eigenvector1);
w=imag(critical_eigenvector1);
theta=0;
epsilon=(tolerance/(length(X0)-1))^0.5; # recall that we defined a tolerance and an X0 in the "first_equilibrium_branch.jl" file 
perturbation=epsilon+1e-12;
x0_time=perturbation*(v*cos(theta)+w*sin(theta))
# next we extract the parameter value at which hopf happens:
rho=bifurcation_points.H.H[end-1,1];
# we then define the system for time integration
f_time(x)=f(x,rho,sigma,beta) # (recall that we defined f and Df in the "my_system.jl" file...)
Df_time(x)=Df(x,rho,sigma,beta)
# finally we define the parameters for the time integration
steps=10^4;
stepsize=0.01;
theta=0.5; # see https://en.wikipedia.org/wiki/Newmark-beta_method
time_evolution=timeseries(x0_time,steps,stepsize,theta,f_time,Df_time);

# finally the moment of truth: we plot our time evolution in phase space and we check whether or not this converges back to the cycle or not
plot(time_evolution.series[1,:],time_evolution.series[2,:])
xlabel!("x")
ylabel!("y")
display(plot!(legend=false))

# conclusion: that's the Lorenz attractor. It's aperiodic, thus not a stable hopf cycle. The first hopf bifurcaiton is then subcritical.
# the user as an excercise can repeat this for the second hopf point. Note that this is bifurcation_points.H.H[:,2] and similarly for the associated eigenvector.