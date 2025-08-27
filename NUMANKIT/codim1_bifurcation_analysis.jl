############################################################################## FUNCTION: LinearizedJacobian ##############################################################################
#                                                                          Linearizes the jacobian of a given system,
#                                                                               Df(u,a)=Df(u,a_0)+a*Df'(u,a_0)
#                                     where a is a continuation parameter. Used as support for the mesp algorithms for the detection of Hopf bifurcations.
# INPUTS
# Fx.............................................................................................................................................function handle to the augmented jacobian
# x_i...............................................................................................ith point on bifurcation diagram (vector containing the continuation parameter, i=1,2)
# OUTPUTS
# A........................................................................................................................the matrix A=Df(u,a_0) (where Df is the non-augmented jacobian)
# B.......................................................................................................................the matrix B=Df'(u,a_0) (where Df is the non-augmented jacobian)
function LinearizedJacobian(Fx,x)
    A=Fx(x);
    dudlambda=matdiv(A[:,1:end-1],A[:,end]);
    dudlambda=[dudlambda;0];
    h1=10^(-8)*norm(x);
    h2=10^(-8)*abs(x[end]);
    directional_component=(Fx(x+h1*dudlambda)-Fx(x-h1*dudlambda))/(2*h1);
    main_component=(Fx([x[1:end-1];x[end]+h2])-Fx([x[1:end-1];x[end]-h2]))/(2*h2);
    B=directional_component+main_component;
    return (A=A[:,1:end-1],B=B[:,1:end-1])
end

#################################################################################### FUNCTION:  mesp1 ####################################################################################
#                                                                         Locates the nearest Hopf bifurcation of
#                                                                                       M*u_t=f(u,lambda)
#                                                                        by looking at the parametrized eigenproblem
#                                                                                    (A+lambda*B)*x=mu*M*x
#                                                                   where (A+lambda*B) is the linearization of Df(u,lambda)
#                                                and performing inverse iteration (the Meerbergen-Spence algorithm) on the matrix eigenproblem
#                                                                          M*Z*A'+A*Z*M'+lambda*(M*Z*B'+B*Z*M')    (1).
# INPUTS
# A.....................................................................................................................................value of the Jacobian around the point of linearisation
# B....................................................................................value of the derivative of the Jacobian wrt the continuation parameter around the point of linearisation
# M..................................................................................................................mass matrix of the problem (optional: if not given it is assumed M=I)
# tol...............................................................................................................tolerance for solving (1), optional: if not given it is assumed tol=10^(-6)
# maxdist.........maximum distance (from the point around which the Jacobian was linearized) of the Hopf bifurcation to be located (optional: if not given it is assumed that maxdist=Inf)
# maxiter...........................................................................................................maximum number of iterations (defaults to Inf, must be greater than 5)
# OUTPUTS
# lambda.....................................................................................................................value of the continuation parameter at which (1) is satisfied
# Z..................................................................................................................................................value of Z for which (1) is satisfied
# mu......................................................................................................................eigenvalue of Df(u,lambda) corresponding to the Hopf bifurcation
# y............................................................................................................................projected eigenvector corresponding to the Hopf bifurcation
function mesp1(A,B;M=eye(size(A,1)),tol=1e-6,maxdist=Inf,maxiter=Inf)
    n=size(A,1);
    V=randn(n,1);
    V=V/norm(V);
    D=1.0;
    Z=sparse(V*D*V');
    lambda=0;
    residual_norm=Inf;
    j=1;
    mu=NaN;
    y=NaN;
    while residual_norm>tol
        Atilde=V'*A*V;
        Btilde=V'*B*V;
        Mtilde=V'*M*V;

        numerator=-tr(Atilde*D*Mtilde'+Mtilde*D*Atilde');
        denominator=tr(Btilde*D*Mtilde'+Mtilde*D*Btilde');
        lambda=numerator/denominator;

        residual=A*Z*M'+M*Z*A'+lambda*(B*Z*M'+M*Z*B');
        residual_norm=norm(residual);

        F=sparse(M*Z*B'+B*Z*M');
        if A isa Float64
            Y=B*Z/A;
        else
            Y=lyapci(real(A)*1.0,real(M)*1.0,-real(F)*1.0)[1];
        end
        Z=sparse(Y/norm(Y));
        V,D=truncated_eigendecomp(Z,2);
        j=j+1;
        if A isa Float64
            mu=1;
            y=1;
        else
            mu,y,_=eigz(V'*(A+lambda*B)*V, Inf, M=V'*M*V);
            mu=mu[1];
        end
        if j>maxiter || (j>5 && abs(lambda)>maxdist)
            break
        end
    end
    if real(mu)<tol
        flag=string("converged in ",j," iterations")
    else
        flag="not converged"
    end
    return (lambda=lambda, Z=Z, mu=mu, y=y, flag=flag)
end

#################################################################################### FUNCTION:  mesp2 ####################################################################################
#                                                                         Locates the nearest Hopf bifurcation of
#                                                                                       M*u_t=f(u,lambda)
#                                                                        by looking at the parametrized eigenproblem
#                                                                                    (A+lambda*B)*x=mu*M*x
#                                                                   where (A+lambda*B) is the linearization of Df(u,lambda)
#                                            and performing inverse iteration (the Meerbergen-Spence algorithm) on the projected matrix eigenproblem
#                                                                          M*Z*A'+A*Z*M'+lambda*(M*Z*B'+B*Z*M')    (1).
# INPUTS
# A................................................................................................................................value of the Jacobian around the point of linearisation
# B...............................................................................value of the derivative of the Jacobian wrt the continuation parameter around the point of linearisation
# M..................................................................................................................mass matrix of the problem (optional: if not given it is assumed M=I)
# k..........................................................................................................................................................size of the projected problem
# tol..........................................................................................................tolerance for solving (1), optional: if not given it is assumed tol=10^(-6)
# maxdist.........maximum distance (from the point around which the Jacobian was linearized) of the Hopf bifurcation to be located (optional: if not given it is assumed that maxdist=Inf)
# maxiter...........................................................................................................maximum number of iterations (defaults to Inf, must be greater than 5)
# OUTPUTS
# lambda.....................................................................................................................value of the continuation parameter at which (1) is satisfied
# Z..................................................................................................................................................value of Z for which (1) is satisfied
# mu......................................................................................................................eigenvalue of Df(u,lambda) corresponding to the Hopf bifurcation
# y............................................................................................................................projected eigenvector corresponding to the Hopf bifurcation
function mesp2(A,B;M=eye(size(A,1)),k=2,tol=1e-6,maxdist=Inf,maxiter=Inf)
    V=rand(size(A,2));
    V=V/norm(V);
    residual_norm=Inf;
    j=0;
    F=0;
    mu=NaN;
    y=NaN;
    lambda=NaN;
    Z=NaN*ones(size(A));
    while residual_norm>tol
        Atilde=V'*A*V;
        Btilde=V'*B*V;
        Mtilde=V'*M*V;
        lambda,Ztilde,_=mesp1(Atilde,Btilde,M=Mtilde,tol=tol,maxdist=maxdist,maxiter=maxiter);
        Z=V*Ztilde*V';
        if j>1
            mu,y,_=eigz(V'*(A+lambda*B)*V, Inf, M=V'*M*V);
            mu=mu[1];
            residual_norm=norm((A+lambda*B)*V*y-mu*M*V*y);
        end
        F=-(B*Z*M'+M*Z*B');
        Y=lyapci(real(A)*1.0,real(M)*1.0,real(F)*1.0)[1];
        V,Dtilde=truncated_eigendecomp(Y,k);
        V=V/norm(V);
        j=j+1;
        if j>maxiter || (j>5 && abs(lambda)>maxdist)
            break
        end
    end
    if abs(real(mu))<tol
        flag=string("converged in ",j," iterations")
    else
        flag="not converged"
    end
    return (lambda=lambda,Z=Z,mu=mu,y=y,flag=flag)
end

############################################################################ FUNCTION: locate_zero_eigenvalue ############################################################################
#                                                                     Locates the nearest Fold bifurcation/Branchpoint of
#                                                                                   M*u_t=f(u,lambda)    (1)
#                                                   by applying newton to a minimally augmented system (see min_aug_system_zero_eigval(...))
# INPUTS
# x0..................a starting point on an equilibrium manifold near a point where the non-augmented jacobian is singular (if chosen too far the minimally augmented system is singular)
# F................................................................................................................the augmented version of the RHS of (1): F(X)=f(u,lambda), X=[u;lambda]
# Fx.......................................................................................................the augmented Jacobian: Fx(X)=[Df(u,lambda) df(u,lambda)/dlambda], X=[u;lambda]
# M......................................................................................................................................................the mass matrix of the system (1)
# tol............................................................................................................tolerance of the method, optional: if not given it is assumed tol=10^(-6)
# maxiter..........................................................................................................................maximum number of iterations, optional: defaults to 100
# OUTPUTS
# X...............................................fold point on the bifurcation diagram X=[x*;lambda*] where x* is an equilibrium of (1) and lambda is the value at which the fold happens
# mu.....................................................................................................................the eigenvalue of smallest magnitude of system (1) at the point X
# flag...................................................................................................a string telling you if the method has converged and if so in how many iterations
function locate_zero_eigenvalue(x0,F,Fx;M=eye(length(x0)-1),tol=1e-6,maxiter=100)
    Ftilde(x)=min_aug_system_zero_eigval(x,F,Fx;M=M);
    Fxtilde(x)=min_aug_jacobian_zero_eigval(x,F,Fx;M=M);
    if abs(eigz(Fxtilde(x0),0)[1][1])<1e-16 || any(isnan.(Fxtilde(x0)))
        return (X=NaN*ones(length(x0)), mu=NaN, flag= "not converged")
    end
    X=newton(x0,Ftilde,Fxtilde,tol,maxiter);
    flag=X.flag;
    X=X.x;
    accu=eigz(Fx(X)[:,1:end-1],0,M=M);
    return (X=X, mu=accu, flag=flag)
end

################################################################################# FUNCTION:  detect_fold #################################################################################
#                                                                        Detects foldpoints in an equilibrium manifold
# INPUTS
# Branch..................................................................................................................A matrix whose columns are equilibria on an equilibrium manifold
# OUTPUTS
# .approximate_foldpoints.....................................................................................A matrix whose columns are approximate foldpoints of an equilibrium manifold
# .flag........................................................................................A string telling the user if the method found any approximate foldpoints and if so how many
function detect_fold(Branch)
    dlambda=Branch[end,2:end]-Branch[end,1:end-1];
    indexes_of_folds_detected=0;
    approximate_foldpoints=0;
    flag="none";
    for i=2:length(dlambda)
        if sign(dlambda[i]*dlambda[i-1])==-1;
            indexes_of_folds_detected=[indexes_of_folds_detected;i];
        end
    end
    if indexes_of_folds_detected isa Vector
        indexes_of_folds_detected=indexes_of_folds_detected[2:end];
        approximate_foldpoints=Branch[:,indexes_of_folds_detected];
        flag=string(length(indexes_of_folds_detected), " folds points found");
    end
    return (approximate_foldpoints=approximate_foldpoints,flag=flag)
end

############################################################################## FUNCTION: detect_branchpoint ##############################################################################
#                                                   Detects possible branchpoints in an equilibrium manifold (these could also be fold points)
# INPUTS
# Branch..................................................................................................................A matrix whose columns are equilibria on an equilibrium manifold
# Fx....................................................................The augmented jacobian (for continuation of equilibria) of the system whose equilibrium manifold is to be analysed
# M.....................................................................................................................The mass matrix of the system (optional, if not given assumes M=I)
# OUTPUTS
# .possible_branchpoints..................................................-........................A matrix whose columns are approximate possible branchpoints of an equilibrium manifold
# .flag.........................................................................................A string telling the user if the method found any possible branchpoints and if so how many
function detect_branchpoint(Branch,Fx;M=eye(length(Branch[1:end-1,1])))
    indexes_of_possible_branchpoints=0;
    possible_branchpoints=0;
    flag="none";
    old_eigval=eigz(Fx(Branch[:,1])[:,1:end-1],0,M=M)[1][1];
    sign_old_eigval=sign(real(old_eigval));
    for i=2:length(Branch[end,:])
        new_eigval=eigz(Fx(Branch[:,i])[:,1:end-1],0,M=M)[1][1];
        sign_new_eigval=sign(real(new_eigval));
        if sign_old_eigval*sign_new_eigval==-1 && abs(imag(new_eigval))<1
            indexes_of_possible_branchpoints=[indexes_of_possible_branchpoints; i];
        end
        sign_old_eigval=sign_new_eigval;
    end
    if indexes_of_possible_branchpoints isa Vector
        indexes_of_possible_branchpoints=indexes_of_possible_branchpoints[2:end];
        possible_branchpoints=Branch[:,indexes_of_possible_branchpoints];
        flag=string(length(indexes_of_possible_branchpoints), " possible branchpoint(s) detected")
    end
    return (possible_branchpoints=possible_branchpoints,flag=flag)
end

################################################################################# FUNCTION:  locate_hopf #################################################################################
#                                                                           Locates the nearest Hopf bifurcation
#                                                                                   M*u_t=f(u,lambda)    (1)
#                                                   by applying newton to a minimally augmented system (see min_aug_system_hopf(...))
# INPUTS
# x0................................................................................................................a starting point on an equilibrium manifold near a presumed hopf point
# F................................................................................................................the augmented version of the RHS of (1): F(X)=f(u,lambda), X=[u;lambda]
# Fx.......................................................................................................the augmented Jacobian: Fx(X)=[Df(u,lambda) df(u,lambda)/dlambda], X=[u;lambda]
# M......................................................................................................................................................the mass matrix of the system (1)
# tol............................................................................................................tolerance of the method, optional: if not given it is assumed tol=10^(-6)
# maxiter..........................................................................................................................maximum number of iterations, optional: defaults to 100
# OUTPUTS
# X...............................................fold point on the bifurcation diagram X=[x*;lambda*] where x* is an equilibrium of (1) and lambda is the value at which the fold happens
# mu......................................................................................................................................the hopf eigenvalue of system (1) at the point X
# vector...........................................the eigenvector of the jacobian of system (1) associated with the hopf eigenvalue (can be used to determine the stability of the cycle)
# flag...................................................................................................a string telling you if the method has converged and if so in how many iterations
function locate_hopf(x0,F,Fx;M=eye(length(x0)-2),tol=1e-6,maxiter=100)
    Ftilde(x)=min_aug_system_hopf(x,F,Fx;M=M);
    Fxtilde(x)=min_aug_jacobian_hopf(x,F,Fx;M=M);
    X=newton(x0,Ftilde,Fxtilde,tol,maxiter);
    flag=X.flag;
    H=X.x;
    omega=H[end];
    X=X.x[1:end-1];
    Df=Fx(X)[:,1:end-1];
    vector=eigz(Df,1.0im*omega,M=M)[2];
    return (H=H, mu=omega, vector=vector, flag=flag)
end

################################################################################## FUNCTION:  find_hopf ##################################################################################
#                                                 Finds approximate hopf points in a system using the equilibrium manifold as initial guess
# INPUTS
# Branch..................................................................................................................A matrix whose columns are equilibria on an equilibrium manifold
# Fx....................................................................The augmented jacobian (for continuation of equilibria) of the system whose equilibrium manifold is to be analysed
# M.....................................................................................................................The mass matrix of the system (optional, if not given assumes M=I)
# tol..........................................................................................................The tolerance with which we want to find the hopf points (defaults to 1e-6)
# maxiter.......................................................................................................The maximum number of iterations for the mesp algorithms (defaults to Inf)
# k...............................................................................................................The size of the projected problem in the mesp2 algorithm (defaults to 2)
# OUTPUTS
# .approx_H..................................................-...........................A matrix whose columns are X_i=[point on the equilibrium manifold; hopf_eigenvalue of this point]
# .flag..........................................................................................A string telling the user if the method found any possible hopf points and if so how many
function find_hopf(Branch,Fx;M=eye(size(Branch,1)-1),tol=1e-6,maxiter=100,k=2)
    avgstepsize=sum(norm.(eachcol(Branch[:,1:end-1]-Branch[:,2:end])))/size(Branch,2);
    avglambdadist=sum(abs.(Branch[end,1:end-1]-Branch[end,2:end]))/size(Branch,2);
    mespBranch=Branch[:,sortperm(Branch[end,:])];
    left_boundary=mespBranch[end,1];
    right_boundary=mespBranch[end,end];
    not_converged_left=false;
    not_converged_right=false;
    reversed=false;
    Mu=zeros(1,0);
    approx_H=zeros(size(Branch,1),0);
    i=0;
    while right_boundary-left_boundary>avgstepsize
        A,B=LinearizedJacobian(Fx,mespBranch[:,1]);
        lambda,Z,mu,y,flag=mesp2(A,B,M=M,k=k,tol=tol,maxdist=abs(mespBranch[end,1]-mespBranch[end,end]),maxiter=maxiter);
        alpha=lambda+mespBranch[end,1];
        if flag!="not converged" && alpha<right_boundary && alpha>left_boundary
            i=i+1;
            alpha_distances=abs.(mespBranch[end,:] .- alpha);
            approx_H=[approx_H mespBranch[:,argmin(alpha_distances)]];
            true_distances=norm.(eachcol(mespBranch .- mespBranch[:,argmin(alpha_distances)]));
            Mu=[Mu mu];
            mespBranch=mespBranch[:,argmin(alpha_distances):end];
            for j=1:size(mespBranch,2)
                if norm(approx_H[:,i]-mespBranch[:,j])>avgstepsize && abs(approx_H[end,i]-mespBranch[end,j])<avglambdadist
                    approx_H=[approx_H mespBranch[:,j]];
                    Mu=[Mu mu];
                end
            end
            if reversed==false
                left_boundary=alpha;
            else
                right_boundary=alpha;
            end
        else
            if reversed==false
                not_converged_left=true;
            else
                not_converged_right=true;
            end
        end
        if reversed==false
            mespBranch=mespBranch[:,mespBranch[end,:] .>left_boundary];
            mespBranch=reverse(mespBranch,dims=2);
            reversed=true;
        else
            mespBranch=mespBranch[:,mespBranch[end,:] .<right_boundary];
            mespBranch=reverse(mespBranch,dims=2);
            reversed=false;
        end
        if not_converged_left==true && not_converged_right==true
            break
        end
    end
    if size(approx_H,2)==0
        flag="not converged";
    else
        tokeep=trues(size(approx_H,2));
        for j=1:size(approx_H,2)
            for l=j+1:size(approx_H,2)
                if norm(approx_H[:,l]-approx_H[:,j])<=avgstepsize
                    tokeep[l]=false;
                end
            end
        end
        approx_H=approx_H[:,tokeep];
        Mu=Mu[tokeep];
        approx_H=[approx_H;imag(Mu)'];
        flag=string(size(approx_H,2), " approximate hopf points found");
    end
    return (approx_H=approx_H,flag=flag)
end

########################################################################## FUNCTION:  get_quadratic_coefficient ##########################################################################
#                                                             Near a Fold bifurcation, a dynamical system can be written as
#                                                                                   x'=lambda+b*x^2
#                                                                where lambda is the bifurcation/cotinuation parameter,
#                                                        and b is a coefficient involved in the (non)-degeneracy of the above
#                                                        This function provides a computation of the quadratic coefficient b.
# INPUTS
# F.....................................................................................................................the augmented version of the RHS of F(X)=f(u,lambda), X=[u;lambda]
# Fx.......................................................................................................the augmented Jacobian: Fx(X)=[Df(u,lambda) df(u,lambda)/dlambda], X=[u;lambda]
# M........................................................................................................................the mass matrix of the system (defaults to the identity matrix)
# LP.................................................................................the fold point H=[state; bifurcation parameter value] whose quadratic coefficient is to be calculated
# OUTPUTS
# l1...................................................................................................................................................the estimated quadratic coefficient
function get_quadratic_coefficient(F,Fx,LP;M=eye(length(LP)-1))
    lambda=LP[end];
    X=LP[1:end-1];
    A=Fx([X;lambda])[:,1:length(X)];
    q=matdiv(A,X);
    p=matdiv(A',X);
    qe=zeros(length(q));
    pe=qe;
    qe[argmax(abs.(q))]=1;
    pe[argmax(abs.(p))]=1;
    qe=sparse(qe);
    pe=sparse(pe);
    q=([A qe; qe' 0]\[zeros(length(q));1])[1:end-1];
    p=([A pe; pe' 0]\[zeros(length(p));1])[1:end-1];
    p=q';
    q=q/norm(q);
    p=p/dot(p,q);
    B(u,v)=(dirder(x -> F([x;lambda]),X,u+v,2)-dirder(x -> F([x;lambda]),X,u-v,2))*0.25
    b=0.5*dot(p,B(q,q));
    return b
end

########################################################################### FUNCTION: get_lyapunov_coefficient ###########################################################################
#                                                          Near a Hopf bifurcation, a dynamical system can be written as
#                                                                             z'=(lambda+i*omega)*z+c*z*|z|^2
#                                                        where z is the amplitude, lambda is the bifurcation/cotinuation parameter,
#                                                               omega is the imaginary component of the critical eigenvalue
#                                                  and c=l1+i*omega1 where omega1 determines how the frequency changes with the amplitude
#                                                     and l1 is referred to as the "first lyapunov coefficient" and determines whether
#                                                     the cycles born at the hopf bifurcation are stable (l1<0, => supercritical hopf)
#                                                         or unstable (l1>0, => subcritical hopf). The case l1=0 is degenerate and
#                                                                        indicates a generalized hopf bifurcation.
#                                                        This function provides a computation of the first lyapunov coefficient l1.
#
#            NOTE: the calculation is approximate and not always reliable for classification purposes (it is only reliable enough to detect a generalized hopf bifurcation),
#                                                      I recommend checking dynamical behaviour via integration to confirm the output.
# INPUTS
# F.....................................................................................................................the augmented version of the RHS of F(X)=f(u,lambda), X=[u;lambda]
# Fx.......................................................................................................the augmented Jacobian: Fx(X)=[Df(u,lambda) df(u,lambda)/dlambda], X=[u;lambda]
# M........................................................................................................................the mass matrix of the system (defaults to the identity matrix)
# H......................................the hopf point H=[state; bifurcation parameter value; imaginary part of critical eigenvalue] whose first lyapunov coefficient is to be calculated
# OUTPUTS
# l1..............................................................................................................................................the estimated first lyapunov coefficient
function get_lyapunov_coefficient(F,Fx,H;M=eye(length(H)-2))
    omega=abs(H[end]);
    lambda=H[end-1];
    X=H[1:end-2];
    A=Fx([X;lambda])[:,1:length(X)];
    q=matdiv(A-M*1.0im+eye(length(X))*eps(Float64),X)
    p=matdiv(A'+M'*1.0im+eye(length(X))*eps(Float64),X)
    qe=zeros(length(X));
    pe=qe;
    idx1=argmax(abs.(q))
    idx2=argmax(abs.(p))
    qe[idx1]=1;
    pe[idx2]=1;
    qe=sparse(qe);
    pe=sparse(pe);
    q=([A-M*1.0im*omega qe; qe' 0]\[zeros(length(X));1])[1:end-1]
    p=([A'+M'*1.0im*omega pe; pe' 0]\[zeros(length(X));1])[1:end-1]
    q=q/norm(q);
    p=p/dot(p,q);
    B(u,v)=(dirder(x -> F([x;lambda]),X,u+v,2)-dirder(x -> F([x;lambda]),X,u-v,2))*0.25
    C(u,v,w)=(dirder(x -> F([x;lambda]),X,u+v+w,3)-dirder(x -> F([x;lambda]),X,u+v,3)-dirder(x -> F([x;lambda]),X,v+w,3)-dirder(x -> F([x;lambda]),X,u+w,3)+dirder(x -> F([x;lambda]),X,u,3)+dirder(x -> F([x;lambda]),X,v,3)+dirder(x -> F([x;lambda]),X,w,3))/48;
    h11=-A\B(q,conj(q));
    h20=(2.0im*omega*M-A)\B(q,q);
    l1=real(dot(p,C(q,q,conj(q))+2*B(q,h11)+B(conj(q),h20)))/(2*omega);
    return l1
end

################################################################################ FUNCTION: analyse_branch ################################################################################
#                                                          Performs exact location of Hopf, Fold and Branch points (in parallel*),
#                                                             also returns the eigenvectors associated with the Hopf eigenvalues
#                                                                 (which can be used to determine the stability of the cycle)
#                                                          and the tangent vectors of the equilibrium manifold at the Branch points
#                                 (the tangent in the direction of the new branch can be determined using the branch point itself and the associated tangent)
#
#                                            *NOTE: the function alone won't run things in parallel unless you add new working processors and export
#                                       the necessery information to these processors. For notes on how to do this, open the file "parallel_env_start.jl"
# INPUTS
# Branch.............................................................................................................................................the equilibrium branch to be analysed
# F.....................................................................................................................the augmented version of the RHS of F(X)=f(u,lambda), X=[u;lambda]
# Fx.......................................................................................................the augmented Jacobian: Fx(X)=[Df(u,lambda) df(u,lambda)/dlambda], X=[u;lambda]
# M........................................................................................................................the mass matrix of the system (defaults to the identity matrix)
# tol.................................................................................the tolerance used in all the methods (the same tolerance will be used everywhere, defaults to 1e-6)
# maxiter..........................................................................................................................maximum number of iterations, optional: defaults to Inf
# hopf......................................................................a boolean telling the function if you want it to check for hopf points (true) or not (false), defaults to true
# fold......................................................................a boolean telling the function if you want it to check for fold points (true) or not (false), defaults to true
# branchpoint.............................................................a boolean telling the function if you want it to check for branch points (true) or not (false), defaults to true
# OUTPUTS
# .H.............................................................................................................................................................a named tuple containing:
# .H.H..................................................................................................................a matrix whose columns are Hopf points on the equilibrium manifold
# .H.V.....................................................................a matrix whose columns are the eigenvectors of the non-augmented jacobian corresponding to the Hopf eigenvalues
# .H.l1............................................................................................a vector containing the first lyapunov coefficients of the Hopf points which were found
# .LP............................................................................................................................................................a named tuple containing:
# .LP.LP................................................................................................................a matrix whose columns are Fold points on the equilibrium manifold
# .LP.b.................................................................................................a vector containing the quadratic coefficients of the Fold points which were found
# .BP............................................................................................................................................................a named tuple containing:
# .BP.BP..............................................................................................................a matrix whose columns are Branch points on the equilibrium manifold
# .BP.tangents..................................................................a matrix whose columns are the tangents to the equilibrium manifold in the direction of the current branch
function analyse_branch(Branch,F,Fx;M=eye(size(Branch,1)-1),tol=1e-6,maxiter=Inf,k=2,hopf=true,fold=true,branchpoint=true)
    if hopf==true
        task1=@spawn begin
            H=zeros(size(Branch,1)+1,0);
            omega=zeros(0);
            eigenvectors=zeros(size(Branch,1)-1,0)
            detected_hopf=find_hopf(Branch,DF;M=M,tol=tol,maxiter=maxiter,k=k)
            if detected_hopf.flag!="not converged"
                approx_H=detected_hopf.approx_H;
                for i=1:size(approx_H,2)
                    reply=locate_hopf(approx_H[:,i],F,Fx,M=M,tol=tol,maxiter=maxiter)
                    H=[H reply.H];
                    omega=[omega; reply.mu]
                    eigenvectors=[eigenvectors reply.vector]
                end
                tokeep=trues(size(H,2))
                for i=1:size(H,2)
                    for j=i+1:size(H,2)
                        if norm(H[1:end-1,i]-H[1:end-1,j])<tol
                            tokeep[j]=false
                        end
                    end
                end
                H=H[:,tokeep]
                omega=omega[tokeep]
                eigenvectors=eigenvectors[:,tokeep]
            end
            l1=zeros(size(H,2));
            for i=1:size(H,2)
                l1[i]=get_lyapunov_coefficient(F,Fx,H[:,i];M=M);
            end
            return (H=H,V=eigenvectors,omega=omega,l1=l1)
        end
    end

    if branchpoint==true
        task2=@spawn begin
            possibleBP,flag=detect_branchpoint(Branch,Fx,M=M);
            BP=zeros(size(Branch,1),0);
            tangents=BP;
            if flag!="none"
                for i=1:size(possibleBP,2)
                    dummy=locate_zero_eigenvalue(possibleBP[:,i],F,Fx;M=M,tol=tol,maxiter=maxiter);
                    if dummy.flag=="converged"
                        BP=[BP dummy.X];
                        X=dummy.X;
                        distances=[norm(Branch[:,j]-dummy.X) for j=1:size(Branch,2)];
                        closest_indices=partialsortperm(distances,1:2);
                        dummy2=Branch[:,closest_indices];
                        tangents=[tangents dummy2[:,2]-dummy2[:,1]];
                    end
                    to_remove=falses(size(BP,2));
                    for i=1:size(BP,2)
                        for j=i+1:size(BP,2)
                            if !to_remove[i] && !to_remove[j]
                            distance=norm(BP[:,i]-BP[:,j])
                                if distance<tol
                                    to_remove[j]=true;
                                end
                            end
                        end
                    end
                end
                return (BP=BP,tangents=tangents)
            else
                return (BP=BP,tangents=tangents)
            end
        end
    end

    if fold==true || branchpoint==true
        task3=@spawn begin
            approxLP,flag=detect_fold(Branch);
            LP=zeros(size(Branch,1),0);
            if flag!="none"
                for i=1:size(approxLP,2)
                    dummy=locate_zero_eigenvalue(approxLP[:,i],F,Fx,M=M,tol=tol,maxiter=maxiter);
                    if dummy.flag!="not converged"
                        LP=[LP dummy.X];
                    end
                end
            end
            b=zeros(size(LP,2))
            for i=1:size(LP,2)
                b[i]=get_quadratic_coefficient(F,Fx,LP[:,i];M=M)
            end
            return (LP=LP,b=b)
        end
    end

    if hopf==true
        H=fetch(task1);
    else
        H=nothing
    end
    if fold==true || branchpoint==true
        LP=fetch(task3);
    else
        LP=nothing
    end
    if branchpoint==true
        BP=fetch(task2);
    else
        BP=nothing
    end

    if branchpoint==true
        pitchfork_indices=falses(size(LP.LP,2))
        for i=1:size(LP.LP,2)
            test=false
            for j=1:size(BP.BP,2)
                if norm(BP.BP[i]-LP[j])<=tol
                    test=true
                end
            end
            if test==false
                pitchfork_indices[i]=true
            end
        end

        indices=trues(size(BP.BP,2))
        for i=1:size(BP.BP,2)
            for j=1:size(LP,2)
                if norm(BP.BP[i]-LP[j])<=tol
                    indices[i]=false
                end
            end
        end
        branchpoints=BP.BP[:,indices]
        tangents=BP.tangents[:,indices]
        pitchforks=LP.LP[:,pitchfork_indices]
        branchpoints=[BP.BP pitchforks]
        pitchfork_tangents=zeros(size(pitchforks))
        for i=1:size(pitchforks,2)
            if size(Branch,1)>3
                pitchfork_tangents[:,i]=[eigs(Fx(pitchforks[:,i])[:,1:end-1],nev=1,which=:SM)[2];0]
            else
                dummy=eigen(Fx(pitchforks[:,i])[:,1:end-1]);
                pitchfork_tangents[:,i]=[dummy.vectors[:,argmin(abs.(dummy.values))];0]
            end
        end
        tangents=[BP.tangents pitchfork_tangents]
        BP=(BP=branchpoints,tangents=tangents)
    end
    return (H=H,LP=LP,BP=BP)
end