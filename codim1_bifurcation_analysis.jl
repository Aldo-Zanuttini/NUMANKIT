################################################################################ FUNCTION: LinearizedJacobian  ################################################################################
#                                                                          Linearizes the jacobian of a given system,
#                                                                               Df(u,a)=Df(u,a_0)+a*Df'(u,a_0)
#                                     where a is a continuation parameter. Used as support for the mesp algorithms for the detection of Hopf bifurcations.
# INPUTS
# Fx..................................................................................................................................................function handle to the augmented jacobian
# x_i....................................................................................................ith point on bifurcation diagram (vector containing the continuation parameter, i=1,2)
# OUTPUTS
# A.............................................................................................................................the matrix A=Df(u,a_0) (where Df is the non-augmented jacobian)
# B............................................................................................................................the matrix B=Df'(u,a_0) (where Df is the non-augmented jacobian)
function LinearizedJacobian(Fx,x)
    A=Fx(x);
    dudlambda=gmres(A[:,1:end-1],A[:,end]);
    dudlambda=[dudlambda;0];
    h1=10^(-8)*norm(x);
    h2=10^(-8)*abs(x[end]);
    directional_component=(Fx(x+h1*dudlambda)-Fx(x-h1*dudlambda))/(2*h1);
    main_component=(Fx([x[1:end-1];x[end]+h2])-Fx([x[1:end-1];x[end]-h2]))/(2*h2);
    B=directional_component+main_component;
    return (A=A[:,1:end-1],B=B[:,1:end-1])
end

####################################################################################### FUNCTION: mesp1 #######################################################################################
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
# M.......................................................................................................................mass matrix of the problem (optional: if not given it is assumed M=I)
# tol...............................................................................................................tolerance for solving (1), optional: if not given it is assumed tol=10^(-6)
# maxdist..............maximum distance (from the point around which the Jacobian was linearized) of the Hopf bifurcation to be located (optional: if not given it is assumed that maxdist=Inf)
# maxiter................................................................................................................maximum number of iterations (defaults to Inf, must be greater than 5)
# OUTPUTS
# lambda..........................................................................................................................value of the continuation parameter at which (1) is satisfied
# Z.......................................................................................................................................................value of Z for which (1) is satisfied
# mu...........................................................................................................................eigenvalue of Df(u,lambda) corresponding to the Hopf bifurcation
# y.................................................................................................................................projected eigenvector corresponding to the Hopf bifurcation
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
            dummy=eigen(Matrix(V'*(A+lambda*B)*V),Matrix(V'*M*V));
            mu=dummy.values[1];
            y=dummy.vectors[:,1];
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

####################################################################################### FUNCTION: mesp2 #######################################################################################
#                                                                         Locates the nearest Hopf bifurcation of
#                                                                                       M*u_t=f(u,lambda)
#                                                                        by looking at the parametrized eigenproblem
#                                                                                    (A+lambda*B)*x=mu*M*x
#                                                                   where (A+lambda*B) is the linearization of Df(u,lambda)
#                                            and performing inverse iteration (the Meerbergen-Spence algorithm) on the projected matrix eigenproblem
#                                                                          M*Z*A'+A*Z*M'+lambda*(M*Z*B'+B*Z*M')    (1).
# INPUTS
# A.....................................................................................................................................value of the Jacobian around the point of linearisation
# B....................................................................................value of the derivative of the Jacobian wrt the continuation parameter around the point of linearisation
# M.......................................................................................................................mass matrix of the problem (optional: if not given it is assumed M=I)
# k...............................................................................................................................................................size of the projected problem
# tol...............................................................................................................tolerance for solving (1), optional: if not given it is assumed tol=10^(-6)
# maxdist..............maximum distance (from the point around which the Jacobian was linearized) of the Hopf bifurcation to be located (optional: if not given it is assumed that maxdist=Inf)
# maxiter................................................................................................................maximum number of iterations (defaults to Inf, must be greater than 5)
# OUTPUTS
# lambda..........................................................................................................................value of the continuation parameter at which (1) is satisfied
# Z.......................................................................................................................................................value of Z for which (1) is satisfied
# mu...........................................................................................................................eigenvalue of Df(u,lambda) corresponding to the Hopf bifurcation
# y.................................................................................................................................projected eigenvector corresponding to the Hopf bifurcation
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
            dummy=eigen(Matrix(V'*(A+lambda*B)*V),Matrix(V'*M*V));
            mu=dummy.values[1];
            y=dummy.vectors[:,1];
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

############################################################################ FUNCTION:  min_aug_system_zero_eigval ############################################################################
#                                                                                 Given a system of the form
#                                                                      Mu_t=f(u,lambda),  Df(u,lambda)=df(u,lambda)/du     (1)
#                                                           and its associated augmented form used for continuation of equilibria
#                                                            Mu_t=F(x),   Fx(x)=[Df(u,lambda), df(u,lambda)/dlambda],   x=(u,lambda)    (2)
#                                                                this function builds a new augmented system from (2) of the form
#                                                                                      G(x)=[F(x);g(x)]        (3)
#                                                                               where g(x) is the solution of
#                                                [Df(u,lambda), p; 0 q']*[w(u,lambda);g(u,lambda)]=[0;1], where Df(u,lambda)*q=Df(u,lambda)'p=0.
#                        this system is nonsingular when Df(u,lambda) is singular. The jacobian is computed in the associated function min_aug_jacobian_zero_eigval.
#                                                           This function is used mainly as support for the locate_fold() function.
# INPUTS
# x......................................................................................................................a point suitable for system (2) at which system (3) is to be evaluated
# F...................................................................................................................................................function handle for the RHS of system (2)
# Fx..................................................................................................................................function handle for the jacobian of the RHS of system (2)
# OUTPUTS
# G......................................................................................................................................the value of the function of system (3) at the point x
function min_aug_system_zero_eigval(x,F,Fx;M=eye(length(x)-1))
    Df=Fx(x)[:,1:end-1];
    n=size(Df,1);
    if n>5
        v=eigs(Df,M,nev=1,which=:SM)[2];
        w=eigs(Df',M',nev=1,which=:SM)[2];
    else
        v=eigen(Df,Matrix(M));
        v=v.vectors[:,argmin(abs.(v.values))];
        w=eigen(Df',Matrix(M));
        w=w.vectors[:,argmin(abs.(w.values))];
    end
    w=w/(w'*v);
    g=[Df w;v' 0]\[zeros(n);1];
    g=g[end];
    G=[F(x);g]
    return G
end

########################################################################### FUNCTION:  min_aug_jacobian_zero_eigval ###########################################################################
#                                                                                 Given a system of the form
#                                                                      Mu_t=f(u,lambda),  Df(u,lambda)=df(u,lambda)/du     (1)
#                                                           and its associated augmented form used for continuation of equilibria
#                                                            Mu_t=F(x),   Fx(x)=[Df(u,lambda), df(u,lambda)/dlambda],   x=(u,lambda)    (2)
#                                                                this function builds the jacobian of the new augmented system
#                                                                                        G(x)=[F(x);g(x)]                      (3)
#                                                             (for more info on this system see: min_aug_system_zero_eigval(...))
#                                                                                 this jacobian has the form
#                                                                                    DG(x)=[Fx(x);dg/dx]
#                                                      where dg/dx=-p'*(dFx/dx)*q and (p,q) are such that Df(u,lambda)*q=Df(u,lambda)'*p
#                                                           This function is used mainly as support for the locate_fold() function.
# INPUTS
# x......................................................................................................................a point suitable for system (2) at which system (3) is to be evaluated
# F...................................................................................................................................................function handle for the RHS of system (2)
# Fx..................................................................................................................................function handle for the jacobian of the RHS of system (2)
# OUTPUTS
# DG.....................................................................................................................................the value of the jacobian of system (3) at the point x
function min_aug_jacobian_zero_eigval(x,F,Fx;M=eye(length(x)-1))
    Df=Fx(x)[:,1:end-1];
    n=size(Df,1);
    if n>5
        v=eigs(Df,M,nev=1,which=:SM)[2];
        w=eigs(Df',M',nev=1,which=:SM)[2];
    else
        v=eigen(Df,Matrix(M));
        v=v.vectors[:,argmin(abs.(v.values))];
        w=eigen(Df',Matrix(M));
        w=w.vectors[:,argmin(abs.(w.values))];
    end
    v=v/(w'*v);
    Bq=(Fx(x+10^(-8)*norm(x)*[v;0])-Fx(x-10^(-8)*norm(x)*[v;0]))/(2*10^(-8)*norm(x));
    gprime=-w'*Bq;
    DG=[Fx(x);gprime]
    return DG
end

############################################################################## FUNCTION:  locate_zero_eigenvalue ##############################################################################
#                                                                     Locates the nearest Fold bifurcation/Branchpoint of
#                                                                                   M*u_t=f(u,lambda)    (1)
#                                                   by applying newton to a minimally augmented system (see min_aug_system_zero_eigval(...))
# INPUTS
# x0.......................a starting point on an equilibrium manifold near a point where the non-augmented jacobian is singular (if chosen too far the minimally augmented system is singular)
# F.....................................................................................................................the augmented version of the RHS of (1): F(X)=f(u,lambda), X=[u;lambda]
# Fx............................................................................................................the augmented Jacobian: Fx(X)=[Df(u,lambda) df(u,lambda)/dlambda], X=[u;lambda]
# M...........................................................................................................................................................the mass matrix of the system (1)
# tol.................................................................................................................tolerance of the method, optional: if not given it is assumed tol=10^(-6)
# maxiter...............................................................................................................................maximum number of iterations, optional: defaults to 100
# OUTPUTS
# X....................................................fold point on the bifurcation diagram X=[x*;lambda*] where x* is an equilibrium of (1) and lambda is the value at which the fold happens
# mu..........................................................................................................................the eigenvalue of smallest magnitude of system (1) at the point X
# flag........................................................................................................a string telling you if the method has converged and if so in how many iterations
function locate_zero_eigenvalue(x0,F,Fx;M=eye(length(x0)-1),tol=1e-6,maxiter=100)
    Ftilde(x)=min_aug_system_zero_eigval(x,F,Fx;M);
    Fxtilde(x)=min_aug_jacobian_zero_eigval(x,F,Fx;M);
    if abs(eigs(Fxtilde(x0),nev=1,which=:SM)[1][1])<1e-16 || any(isnan.(Fxtilde(x0)))
        return (X=NaN*ones(length(x0)), mu=NaN, flag= "not converged")
    end
    X=newton(x0,Ftilde,Fxtilde,tol,maxiter);
    flag=X.flag;
    X=X.x;
    if size(Fx(X),1)<5
        accu=minimum(abs.(eigen(Fx(X)[:,1:end-1]).values));
    else
        accu=eigs(Fx(X)[:,1:end-1],nev=1,which=:SM);
    end
    return (X=X, mu=accu, flag="converged")
end

#################################################################################### FUNCTION: detect_fold ####################################################################################
#                                                                        Detects foldpoints in an equilibrium manifold
# INPUTS
# Branch.......................................................................................................................A matrix whose columns are equilibria on an equilibrium manifold
# OUTPUTS
# .approximate_foldpoints..........................................................................................A matrix whose columns are approximate foldpoints of an equilibrium manifold
# .flag.............................................................................................A string telling the user if the method found any approximate foldpoints and if so how many
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

################################################################################ FUNCTION:  detect_branchpoint ################################################################################
#                                                   Detects possible branchpoints in an equilibrium manifold (these could also be fold points)
# INPUTS
# Branch.......................................................................................................................A matrix whose columns are equilibria on an equilibrium manifold
# Fx.........................................................................The augmented jacobian (for continuation of equilibria) of the system whose equilibrium manifold is to be analysed
# M..........................................................................................................................The mass matrix of the system (optional, if not given assumes M=I)
# OUTPUTS
# .possible_branchpoints.......................................................-........................A matrix whose columns are approximate possible branchpoints of an equilibrium manifold
# .flag..............................................................................................A string telling the user if the method found any possible branchpoints and if so how many
function detect_branchpoint(Branch,Fx;M=eye(length(Branch[1:end-1],1)))
    indexes_of_possible_branchpoints=0;
    possible_branchpoints=0;
    flag="none";
    if length(Branch[:,1])>4
        old_eigval=eigs(Fx(Branch[:,1])[:,1:end-1],nev=1,which=:SM)[1][1];
        old_eigval=sign(real(old_eigval));
    else
        old_eigval=eigen(Fx(Branch[:,1])[:,1:end-1]).values[argmin(abs.(eigen(Fx(Branch[:,1])[:,1:end-1]).values))];
        old_eigval=sign(real(old_eigval));
    end
    for i=2:length(Branch[end,:])
        if length(Branch[:,1])>4
            new_eigval=eigs(Fx(Branch[:,i])[:,1:end-1],nev=1,which=:SM)[1][1];
            new_eigval=sign(real(new_eigval));
        else
            new_eigval=eigen(Fx(Branch[:,i])[:,1:end-1]).values[argmin(abs.(eigen(Fx(Branch[:,i])[:,1:end-1]).values))];
            new_eigval=sign(real(new_eigval));
        end
        if old_eigval*new_eigval==-1
            indexes_of_possible_branchpoints=[indexes_of_possible_branchpoints; i];
        end
        old_eigval=new_eigval;
    end
    if indexes_of_possible_branchpoints isa Vector
        indexes_of_possible_branchpoints=indexes_of_possible_branchpoints[2:end];
        possible_branchpoints=Branch[:,indexes_of_possible_branchpoints];
        flag=string(length(indexes_of_possible_branchpoints), " possible branchpoint(s) detected")
    end
    return (possible_branchpoints=possible_branchpoints,flag=flag)
end

##################################################################################### FUNCTION: find_hopf #####################################################################################
#                                                 Finds approximate hopf points in a system using the equilibrium manifold as initial guess
# INPUTS
# Branch.......................................................................................................................A matrix whose columns are equilibria on an equilibrium manifold
# F......................................................................The augmented RHS function (for continuation of equilibria) of the system whose equilibrium manifold is to be analysed
# Fx.........................................................................The augmented jacobian (for continuation of equilibria) of the system whose equilibrium manifold is to be analysed
# M..........................................................................................................................The mass matrix of the system (optional, if not given assumes M=I)
# tol...............................................................................................................The tolerance with which we want to find the hopf points (defaults to 1e-6)
# maxiter.............................................................................................................The maximum number of iterations for the mesp algorithm (defaults to Inf)
# k.....................................................................................................................The size of the projected problem in the mesp algorithm (defaults to 2)
# old_hopf....................................................................Don't touch this: the function is recursive and needs input from its previous iteration, this is what old_hopf is
# OUTPUTS
# .possible_hopf.......................................................-........................................A matrix whose columns are approximate possible hopf points of the given system
# .flag...............................................................................................A string telling the user if the method found any possible hopf points and if so how many
function find_hopf(Branch,F,Fx;M=eye(size(Branch[1:end-1,1],1)),tol=1e-6,maxiter=Inf,k=2,old_hopf=ones(size(M,1)+1)*NaN)
    if size(Branch,2)>0
        A,B=LinearizedJacobian(Fx,Branch[:,1]);
        dummy=mesp2(A,B,M=M,k=k,tol=tol,maxdist=norm(Branch[:,1]-Branch[:,end]),maxiter=maxiter);
        flag1=dummy.flag;
        NewBranch=Branch;
        if flag1!="not converged"
            lambda=Branch[end,1]+dummy.lambda;
            guess=Branch[:,argmin(abs(Branch[end,i]-lambda) for i=1:size(Branch,2))];
            possible_hopf=init_cont(lambda,F,Fx,guess[1:end-1]).X0;
            mu=dummy.mu;
            RCSS=norm(Branch[:,1]-possible_hopf);
            distances=[norm(Branch[:,i]-possible_hopf) for i=1:size(Branch,2)];
            NewBranch=Branch[:,sortperm(distances)];
            dummy=find_hopf(NewBranch,F,Fx;M=M,tol=tol,maxiter=maxiter,k=k,old_hopf=possible_hopf[:,end]);
            if norm(dummy.possible_hopf)<Inf
                possible_hopf=[possible_hopf dummy.possible_hopf];
                mu=[mu dummy.mu];
                lambda=[lambda dummy.lambda];
            end
            flag2=string(size(possible_hopf,2)," possible hopf points found");
        else
            flag2="not converged";
            possible_hopf=NaN*ones(size(Branch,1));
            mu=NaN;
            lambda=NaN;
        end
        return (possible_hopf=possible_hopf, mu=mu, lambda=lambda, flag=flag2)
    else
        return (possible_hopf=NaN*ones(size(Branch,1)), mu=NaN, lambda=NaN, flag="that's it!")
    end
end

################################################################################## FUNCTION:  analyse_branch ##################################################################################
#                                                          Performs exact location of Hopf, Fold and Branch points (in parallel*),
#                                                             also returns the eigenvectors associated with the Hopf eigenvalues
#                                                                 (which can be used to determine the stability of the cycle)
#                                                          and the tangent vectors of the equilibrium manifold at the Branch points
#                                 (the tangent in the direction of the new branch can be determined using the branch point itself and the associated tangent)
#
#                                            *NOTE: the function alone won't run things in parallel unless you add new working processors and export
#                                       the necessery information to these processors. For notes on how to do this, open the file "parallel_env_start.jl"
# INPUTS
# Branch..................................................................................................................................................the equilibrium branch to be analysed
# F..........................................................................................................................the augmented version of the RHS of F(X)=f(u,lambda), X=[u;lambda]
# Fx............................................................................................................the augmented Jacobian: Fx(X)=[Df(u,lambda) df(u,lambda)/dlambda], X=[u;lambda]
# M.............................................................................................................................the mass matrix of the system (defaults to the identity matrix)
# tol......................................................................................the tolerance used in all the methods (the same tolerance will be used everywhere, defaults to 1e-6)
# maxiter...............................................................................................................................maximum number of iterations, optional: defaults to Inf
# hopf...........................................................................a boolean telling the function if you want it to check for hopf points (true) or not (false), defaults to true
# fold...........................................................................a boolean telling the function if you want it to check for fold points (true) or not (false), defaults to true
# branchpoint..................................................................a boolean telling the function if you want it to check for branch points (true) or not (false), defaults to true
# OUTPUTS
# .H..................................................................................................................................................................a named tuple containing:
# .H.H.......................................................................................................................a matrix whose columns are Hopf points on the equilibrium manifold
# .H.V..........................................................................a matrix whose columns are the eigenvectors of the non-augmented jacobian corresponding to the Hopf eigenvalues
# .LP........................................................................................................................a matrix whose columns are Fold points on the equilibrium manifold
# .BP.................................................................................................................................................................a named tuple containing:
# .BP.BP...................................................................................................................a matrix whose columns are Branch points on the equilibrium manifold
# .BP.tangents.......................................................................a matrix whose columns are the tangents to the equilibrium manifold in the direction of the current branch
function analyse_branch(Branch,F,Fx;M=eye(size(Branch[1:end-1,1],1)),tol=1e-6,maxiter=Inf,k=2,hopf=true,fold=true,branchpoint=true)
    if hopf==true
        task1=@spawn begin
            H=zeros(size(Branch,1),0);
            eigenvectors=zeros(size(Branch,1)-1,0)
            detected_hopf=find_hopf(Branch,F,DF;M=M,tol=tol^0.5,maxiter=maxiter,k=k)
            if detected_hopf.flag!="not converged"
                n=size(detected_hopf.possible_hopf,2);
                toremove=falses(n);
                for i=1:n
                    for j=(i+1):n
                        if !toremove[i] && !toremove[j]
                            distance=norm(detected_hopf.possible_hopf[:,i]-detected_hopf.possible_hopf[:,j])
                            if distance<tol
                                toremove[j]=true;
                            end
                        end
                    end
                end
                detected_hopf=(flag=detected_hopf.flag, detected_hopf=detected_hopf.possible_hopf[:,.!toremove],mu=detected_hopf.mu);
                for i=1:size(detected_hopf.detected_hopf,2)
                    param_value=detected_hopf.detected_hopf[end,i];
                    A,B=LinearizedJacobian(DF,detected_hopf.detected_hopf[:,i]);
                    dummy2=mesp2(A,B;M=M,k=k,tol=tol);
                    if dummy2.flag!="not converged"
                        dummy=dummy2;
                        param_value=param_value+dummy.lambda;
                    else
                        dummy=detected_hopf;
                        param_value=detected_hopf.detected_hopf[end,i];
                    end
                    ftilde(x)=F([x;param_value]);
                    Dftilde(x)=DF([x;param_value])[:,1:end-1];
                    corrected_hopf=newton(detected_hopf.detected_hopf[1:end-1,i],ftilde,Dftilde,tol,maxiter);
                    matrix=Dftilde(corrected_hopf.x);
                    if size(matrix,1)>3
                        eigenvectors=[eigenvectors eigs(Dftilde(corrected_hopf.x),sigma=dummy.mu,nev=1,which=:LM)[2]];
                    else
                        dummy3=eigen(matrix);
                        index=argmin(abs.(real.(dummy3.values)));
                        eigenvectors=[eigenvectors dummy3.vectors[:,index]];
                    end
                    H=[H [corrected_hopf.x;param_value]];
                end
                return (H=H,eigenvectors=eigenvectors)
            else
                return (H=NaN,eigenvectors=NaN)
            end
        end
    end
    if fold==true
        task2=@spawn begin
            approxLP,flag=detect_fold(Branch);
            LP=zeros(size(Branch,1),0);
            if flag!="none"
                for i=1:size(approxLP,2)
                    dummy=locate_zero_eigenvalue(approxLP[:,i],F,Fx,M=M,tol=tol,maxiter=maxiter);
                    if dummy.flag=="converged"
                        LP=[LP dummy.X];
                    end
                end
            end
            return LP
        end
    end
    if branchpoint==true
        task3=@spawn begin
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
                return (BP=NaN,tangents=NaN)
            end
        end
    end
    if hopf==true
        H=fetch(task1);
    else
        H=nothing
    end
    if fold==true
        LP=fetch(task2);
    else
        LP=nothing
    end
    if branchpoint==true
        BP=fetch(task3);
    else
        BP=nothing
    end
    return (H=H,LP=LP,BP=BP)
end