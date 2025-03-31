######################## FUNCTION: init_cont #########################
# gives an initial point on an equilibrium branch, together with the
# tangent vector to said point
# INPUTS
# initial_param_value..............value of the bifurcation parameter
#                                  at the first point on the
#                                  equilibrium manifold
# F................................augmented RHS function, i.e.
#                                       x'=f(x,a)=F(X)
#                                  where X=[x;a]
# Fx...............................augmented Jacobian
# initial_guess....................a random vector of length
#                                  length(X)-1 (equiv. of length 
#                                  length(x), i.e. the state without
#                                  the bifurcation parameter)
# OUTPUTS a named tuple with names
# .X0..............................first point on equilibrium manifold
# .v0..............................tanget to equilibrium manifold at
#                                  X0
function init_cont(init_param_value, F, Fx,initial_guess)
    g(x)=F([x;init_param_value])
    gx(x)=Fx([x;init_param_value])[:,1:end-1]
    X_init=newton(initial_guess,g,gx,10^(-12),10^6).x
    h(x)=F([X_init;init_param_value+0.1])
    hx(x)=Fx([X_init;init_param_value+0.1])[:,1:end-1]
    V_init=X_init-newton(X_init,h,hx,10^(-12),10^6).x
    V_init=[V_init;init_param_value-0.1];
    V_init=V_init/norm(V_init)
    return (X0=[X_init;init_param_value], v0=V_init)
end
########################### FUNCTION: cont ###########################
# performs numerical continuation of the equilibria/bifurcations of a
# (infinite dimensional) dynamical system of the form
#                           Mx'=f(x,a)=F(X)
# using a modified (up to 5th order) Moore-Penrose scheme
# INPUTS
# initstepsize...................initial stepsize for the prediction
#                                step
# maxstepsize....................maximum stepsize allowed for the
#                                prediction step
# maxiter........................maximum number of corrections allowed
# steps..........................number of points to compute on the
#                                equilibrium/bifurcation branch
# ord............................order of accuracy of the corrections
#                                (2<=ord<=5), for ord>2 derivatives
#                                will be calculated numerically
#                                (see dirder() function in the
#                                basic_tools.jl file)
# F..............................the RHS of the dynamical system
# Fx.............................the Jacobian Matrix of the dynamical
#                                system
# X..............................the first point on the equilibrium/
#                                bifurcation branch
# v..............................the tangent to the first point on the
#                                equilibrium/bifurcation branch
# dir............................the direction of the continuation
#                                (+1 to move in the positive
#                                continuation parameter direction, -1
#                                to move in the negative continuation
#                                parameter direction)
# OUTPUTS
# Branch.........................a matrix whose columns are points
#                                on the bifurcation/equilibrium branch
function cont(initstepsize,maxstepsize,maxiter,steps,ord,F,Fx,X,v,dir)
    v=v/norm(v);
    vpred=v;
    Branch=zeros(length(X),steps+1);
    Branch[:,1]=X;
    s=initstepsize;
    n=maxiter;
    for j=1:steps
        Xpred=X+s*dir*v;
        if n<maxiter/2 && s<=maxstepsize/2
            s=s*2;
        end
        n=0;
        while norm(F(Xpred))>10^(-12) || norm(Xpred-X)>maxstepsize || vpred'*v<0
            if ord==2
                Xpred=Xpred - mpinv(Fx(Xpred),F(Xpred));
            elseif ord==3
                Fxx=dirder(Fx,Xpred,Xpred,1)*Xpred;
                Xpred=Xpred - mpinv(Fx(Xpred),F(Xpred)) - (mpinv(Fx(Xpred),F(Xpred))).^2 .* (mpinv(Fx(Xpred),Fxx))/2;
            elseif ord==4
                Fxx=dirder(Fx,Xpred,Xpred,1)*Xpred;
                Fxxx=dirder(Fx,Xpred,Xpred,2)*Xpred;
                Xpred=Xpred - mpinv(Fx(Xpred),F(Xpred)) - (mpinv(Fx(Xpred),F(Xpred))).^2 .* (mpinv(Fx(Xpred),Fxx))/2 - (mpinv(Fx(Xpred),F(Xpred))).^3 .* ((mpinv(Fx(Xpred),Fxx)).^2/2-mpinv(Fx(Xpred),Fxxx)/6);
            elseif ord==5
                Fxx=dirder(Fx,Xpred,Xpred,1)*Xpred;
                Fxxx=dirder(Fx,Xpred,Xpred,2)*Xpred;
                Fxxxx=dirder(Fx,Xpred,Xpred,3)*Xpred;
                Xpred=Xpred - mpinv(Fx(Xpred),F(Xpred)) - (mpinv(Fx(Xpred),F(Xpred))).^2 .* (mpinv(Fx(Xpred),Fxx))/2 - (mpinv(Fx(Xpred),F(Xpred))).^3 .* ((mpinv(Fx(Xpred),Fxx)).^2/2-mpinv(Fx(Xpred),Fxxx)/6) - (mpinv(Fx(Xpred),F(Xpred))).^4 .*(5*(mpinv(Fx(Xpred),Fxx)).^3/8 - mpinv(Fx(Xpred),Fxx) .* mpinv(Fx(Xpred),Fxxx)*5/12+mpinv(Fx(Xpred),Fxxxx)/24);
            end
            vpred=[Fx(Xpred);v']\[zeros(length(Fx(Xpred)[:,1]));1];
            n=n+1;
            if n>maxiter
                s=s/2;
                Xpred=X+s*dir*v;
                n=0;
            end
        end
        X=Xpred;
        Branch[:,j+1]=X;
        v=[Fx(X);v']\[zeros(length(Fx(X)[:,1]));1];
        v=v/norm(v);
    end
    return Branch
end

###################### FUNCTION:  switch_branch ######################
# given a branchpoint X and a tangent/secant V, computes the new
# tangent at X for the cont function.
# INPUTS
# X..............................a branch point
# V..............................a tangent to the bifurcation diagram
#                                (or secant) at x
# Fx.............................function handle to the augmented
#                                jacobian (the one used in cont)
# M..............................the mass matrix of the system
#                                (optional)
# OUTPUTS
# Vnew...........................the tangent at X along the new branch
function switch_branch(X,V,Fx,M=eye(length(X)))
    F=[Fx(X);V'];
    if size(F,1)<5
        Vnew=eigen(F,Matrix(M));
        Vnew=Vnew.vectors[:,argmin(abs.(Vnew.values))];
    else
        Vnew=eigs(F,M,nev=1,which=:SM)[2];
    end
    return real(Vnew)
end
