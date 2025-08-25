##################################################################################### FUNCTION: eigz #####################################################################################
#                                                                       Solves the generalized eigenvalue problem
#                                                                                     Ax=lambda*M*x
#                                                                        using one of the solvers specified below
#                                                                           (direct methods, Arnlodi, JDQZ)
# INPUTS (mandatory)
# A......................................................................................................................................................LHS matrix (typically a Jacobian)
# lambda......................................................................................the target eigenvalue (0=Smallest Maginutude, Inf=Largest Magnitude and any value inbetween)
# INPUTS (optional)
# M..................................................................................................................RHS matrix (typically a mass matrix, defaults to the identity matrix)
# maxiter..............................................................................................the maximum number of iterations (defaults to 100, only used for iterative methods)
# tol.............................................................................................the desired tolerance of the method (defaults to 1e-12, only used for iterative methods)
# howmany.................................................how many eigenvalues to be computed (defaults to 1, note that the last two solvers in this file can only compute one eigenvalue)
# both.........................a boolean telling the method if both the RHS and LHS eigenvectors need to be computed (defaults to false, note that this option is only available for JDQZ)
# maxsubspace................................................................................................................................maximum subspace size for JDQZ and/or HP_JDQZ
# v0...........................................................................an initial guess for the eigenvector, only applies to the high-precision methods (the last two in the file)
# method.......................a string telling eigz() what method should be used (auto, direct, jdqz, arnoldi), defaults to auto. auto chooses the appropriate method based on the 
#                              system's size (JDQZ for large systems, direct for small systems), if the method is iterative (jdqz or arnoldi) eigz() also automatically decides to use the
#                              high-precision methods or the normal precision methods based on the given tolerance.
# OUTPUTS
# a named tuple with names:
# values................................................................................................................................the calculated eigenvalues (NB: this is a vector!)
# vectors.........................................................................................................................(a matrix whose columns are) the calculated eigenvectors
# error.........................................................................................the error with which the eigenvalues/vectors were calculated (always 0 for direct methods)
# converged...........................................................................a flag telling the user if the error is within the tolerance or not (always true for direct methods)
function eigz(A, lambda; M=eye(size(A,1)), maxiter=100, tol=1e-12, howmany=1, both=false, maxsubspace=min(10,size(A,1)), v0=rand(size(A,1)), method="auto")
    dims=size(A,1)
    if method=="auto"
        if dims<=10
            method="direct"
        elseif lambda!=Inf
            method="jdqz"
        else
            method="arnoldi"
        end
    end
    if method=="direct"
        return standardized_eigen(A,lambda,M=M,howmany=howmany)
    end
    if method=="jdqz"
        if tol>=1e-16
            return standardized_jdqz(A,lambda, both=both, howmany=howmany, M=M,maxsubspace=maxsubspace,maxiter=maxiter,tol=tol)
        else
            return HP_jd(A, lambda, v0=v0, M=M, tol=tol, maxiter=maxiter, maxsubspace=maxsubspace)
        end
    elseif method=="arnoldi"
        if tol>=1e-16
            return standardized_eigs(A,lambda, howmany=howmany, M=M, maxiter=maxiter, tol=tol)
        else
            return HP_eigs(A, lambda, v0=v0, M=M,  tol=tol, maxiter=maxiter)
        end
    end
end

############################################################################## FUNCTION: standardized_eigen ##############################################################################
#                                                                       Solves the generalized eigenvalue problem
#                                                                                     Ax=lambda*M*x
#                                                                        using LinearAlgebra's eigs() routine,
#                                                  the output is standardized to match the outputs of all other methods in this file
# INPUTS (mandatory)
# A......................................................................................................................................................LHS matrix (typically a Jacobian)
# lambda......................................................................................the target eigenvalue (0=Smallest Maginutude, Inf=Largest Magnitude and any value inbetween)
# INPUTS (optional)
# M..................................................................................................................RHS matrix (typically a mass matrix, defaults to the identity matrix)
# howmany.................................................how many eigenvalues to be computed (defaults to 1, note that the last two solvers in this file can only compute one eigenvalue)
# OUTPUTS
# a named tuple with names:
# values................................................................................................................................the calculated eigenvalues (NB: this is a vector!)
# vectors.........................................................................................................................(a matrix whose columns are) the calculated eigenvectors
# error.........................................................................................the error with which the eigenvalues/vectors were calculated (always 0 for direct methods)
# converged...........................................................................a flag telling the user if the error is within the tolerance or not (always true for direct methods)
function standardized_eigen(A ,lambda; M=eye(size(A,1)), howmany=1)
    values,vectors=eigen(Matrix(A),Matrix(M))
    if lambda!=Inf
        idx=sortperm(abs.(values .- lambda))[1:howmany]
    else
        idx=reverse(sortperm(abs.(values)))[1:howmany]
    end
    value=values[idx]
    vector=vectors[:,idx]
    return (values=value, vectors=vector, error=0, converged=true)
end

############################################################################## FUNCTION:  standardized_eigs ##############################################################################
#                                                                       Solves the generalized eigenvalue problem
#                                                                                     Ax=lambda*M*x
#                                                                        using eigs() from Julia's ARPACK wrapper
# INPUTS (mandatory)
# A......................................................................................................................................................LHS matrix (typically a Jacobian)
# lambda......................................................................................the target eigenvalue (0=Smallest Maginutude, Inf=Largest Magnitude and any value inbetween)
# INPUTS (optional)
# M..................................................................................................................RHS matrix (typically a mass matrix, defaults to the identity matrix)
# maxiter..............................................................................................the maximum number of iterations (defaults to 100, only used for iterative methods)
# tol.............................................................................................the desired tolerance of the method (defaults to 1e-12, only used for iterative methods)
# howmany.................................................how many eigenvalues to be computed (defaults to 1, note that the last two solvers in this file can only compute one eigenvalue)
# OUTPUTS
# a named tuple with names:
# values................................................................................................................................the calculated eigenvalues (NB: this is a vector!)
# vectors.........................................................................................................................(a matrix whose columns are) the calculated eigenvectors
# error.........................................................................................the error with which the eigenvalues/vectors were calculated (always 0 for direct methods)
# converged...........................................................................a flag telling the user if the error is within the tolerance or not (always true for direct methods)
function standardized_eigs(A, lambda; howmany=1, M=eye(size(A,1)), maxiter=100, tol=1e-12)
    if lambda!=Inf
        dummy=eigs(A-M*lambda,which=:SM,nev=howmany,tol=tol,maxiter=maxiter)
    else
        dummy=eigs(A-M,which=:LM,nev=howmany,tol=tol,maxiter=maxiter)
    end
    value=dummy[1]
    vector=dummy[2]
    nconv=dummy[3]
    error=norm(dummy[6])
    if lambda!=Inf
        values=value .+lambda
    else
        values=value
    end
    if nconv==howmany
        converged=true
    else
        converged=false
    end
    return (vectors=vector,values=value,lambda=lambda,error=error,converged=converged)
end

############################################################################## FUNCTION:  standardized_jdqz ##############################################################################
#                                                                       Solves the generalized eigenvalue problem
#                                                                                     Ax=lambda*M*x
#                             using Harmen Stoppel's and Stefanos Carlström's JacobiDavidson's jdqz() routine (https://github.com/haampie/JacobiDavidson.jl)
# INPUTS (mandatory)
# A......................................................................................................................................................LHS matrix (typically a Jacobian)
# lambda......................................................................................the target eigenvalue (0=Smallest Maginutude, Inf=Largest Magnitude and any value inbetween)
# INPUTS (optional)
# M..................................................................................................................RHS matrix (typically a mass matrix, defaults to the identity matrix)
# maxiter..............................................................................................the maximum number of iterations (defaults to 100, only used for iterative methods)
# tol.............................................................................................the desired tolerance of the method (defaults to 1e-12, only used for iterative methods)
# howmany.................................................how many eigenvalues to be computed (defaults to 1, note that the last two solvers in this file can only compute one eigenvalue)
# both.........................a boolean telling the method if both the RHS and LHS eigenvectors need to be computed (defaults to false, note that this option is only available for JDQZ)
# maxsubspace................................................................................................................................maximum subspace size for JDQZ and/or HP_JDQZ
# OUTPUTS
# a named tuple with names:
# values................................................................................................................................the calculated eigenvalues (NB: this is a vector!)
# vectors.........................................................................................................................(a matrix whose columns are) the calculated eigenvectors
# error.........................................................................................the error with which the eigenvalues/vectors were calculated (always 0 for direct methods)
# converged...........................................................................a flag telling the user if the error is within the tolerance or not (always true for direct methods)
function standardized_jdqz(A,lambda; howmany=1, both=false, M=eye(size(A,1)),maxsubspace=min(10,size(A,1)),maxiter=100,tol=1e-12)
    schur,residuals=jdqz(A,M,pairs=howmany,target=Near(1.0*Complex(lambda)),subspace_dimensions=1:maxsubspace,max_iter=maxiter,tolerance=tol,verbosity=1)
    lambda=(schur.alphas ./schur.betas)
    err=residuals[3]
    if err<tol
        converged=true
    else
        converged=false
    end
    if both==false
        vector=schur.Q.basis
        return (vectors=vector,values=lambda,error=err,converged=converged)
    else
        Lvector=schur.Z.basis
        Rvector=schur.Q.basis
        return (Rvectors=Rvector,Lvectors=Lvector,values=lambda,error=err,converged=converged)
    end
end

#################################################################################### FUNCTION:  HP_jd ####################################################################################
#                                                                       Solves the generalized eigenvalue problem
#                                                                                     Ax=lambda*M*x
#                                                                      using a custom Jacobi Davidson implementation
#                                                                   which supports BigFloat-values matrices and vectors
#                                                                         (and thus allows for O(1e-77) accuracy).
#                                                                     NOTE: only allows for computation of ONE eigenpair!
# INPUTS (mandatory)
# A......................................................................................................................................................LHS matrix (typically a Jacobian)
# lambda......................................................................................the target eigenvalue (0=Smallest Maginutude, Inf=Largest Magnitude and any value inbetween)
# INPUTS (optional)
# M..................................................................................................................RHS matrix (typically a mass matrix, defaults to the identity matrix)
# maxiter..............................................................................................the maximum number of iterations (defaults to 100, only used for iterative methods)
# tol.............................................................................................the desired tolerance of the method (defaults to 1e-12, only used for iterative methods)
# maxsubspace................................................................................................................................maximum subspace size for JDQZ and/or HP_JDQZ
# v0...........................................................................an initial guess for the eigenvector, only applies to the high-precision methods (the last two in the file)
# OUTPUTS
# a named tuple with names:
# values................................................................................................................................the calculated eigenvalues (NB: this is a vector!)
# vectors.........................................................................................................................(a matrix whose columns are) the calculated eigenvectors
# error.........................................................................................the error with which the eigenvalues/vectors were calculated (always 0 for direct methods)
# converged...........................................................................a flag telling the user if the error is within the tolerance or not (always true for direct methods)
function HP_jd(A, lambda; v0=rand(size(A,1)), M=eye(size(A,1)), tol=BigFloat(1e-50), maxiter=1000, maxsubspace=min(10, size(A,1)))
    T = Complex{BigFloat}
    A = sparse(T.(A))
    M = sparse(T.(M))
    lambda = T(lambda)
    V = zeros(T, size(A,1), maxsubspace)
    W = zeros(T, size(A,1), maxsubspace)
    V[:,1] = T.(v0) ./ max(norm(v0), eps(BigFloat))
    W[:,1] = M * V[:,1]
    residuals = zeros(BigFloat, maxsubspace)
    ritz_values = zeros(T, maxsubspace)
    for iter in 1:maxiter
        for k in 1:maxsubspace
            H = W[:,1:k]' * A * V[:,1:k]
            K = W[:,1:k]' * M * V[:,1:k]
            mat=(K+eye(k)*eps(BigFloat))\H
            dummy=schur(mat)
            if istriu(dummy.T)
                dummy3=dummy.Z
            else
                dummy2=eigen(dummy.T)
                dummy3=dummy.Z*dummy2.vectors
            end
            dummy3=dummy3 ./ norm.(eachcol(dummy3))'
            F=(values=dummy.values,vectors=dummy3)
            idx = argmin(abs.(F.values .- lambda))
            theta, y = F.values[idx], F.vectors[:,idx]
            u = V[:,1:k] * (y ./ norm(y))
            r = A*u - theta*M*u
            res = norm(r) 
            residuals[k] = res
            ritz_values[k] = theta
            if res < tol
                u=Float64.(real(u))+im*Float64.(imag(u))
                u=u/norm(u)
                theta=Float64(real(theta))+im*Float64(imag(theta))
                return (vector=u, value=theta, error=res, converged=true)
            end
            if k < maxsubspace
                P=(eye(size(A,1))-M*u*u')*(A - lambda*M )*(eye(size(A,1))-u*u'*M)
                LHS=(eye(size(A,1))-M*u*u')*(A - theta*M )*(eye(size(A,1))-u*u'*M)
                r=(A - theta*M)*u
                t = matdiv(P*LHS, -P*r)
                V[:,k+1] = t ./ norm(t)
                W[:,k+1] = M * V[:,k+1]
            end
        end
        best = partialsortperm(abs.(ritz_values .- lambda), 1:min(3, maxsubspace))
        V = hcat(V[:,best], zeros(T, size(A,1), maxsubspace-length(best)))
        W = hcat(W[:,best], zeros(T, size(A,1), maxsubspace-length(best)))
    end
    best_idx = argmin(residuals)
    u=V[:,best_idx]
    u=Float64.(real(u))+im*Float64.(imag(u))
    u=u/norm(u)
    theta=ritz_values[best_idx]
    theta=Float64(real(theta))+im*Float64(imag(theta))
    return (vector=u, value=theta, error=residuals[best_idx], converged=false)
end

################################################################################### FUNCTION:  HP_eigs ###################################################################################
#                                                                       Solves the generalized eigenvalue problem
#                                                                                     Ax=lambda*M*x
#                                                                      using a custom shift and invert implementation
#                                                                   which supports BigFloat-values matrices and vectors
#                                                                         (and thus allows for O(1e-77) accuracy).
#                                                                     NOTE: only allows for computation of ONE eigenpair!
# INPUTS (mandatory)
# A......................................................................................................................................................LHS matrix (typically a Jacobian)
# lambda......................................................................................the target eigenvalue (0=Smallest Maginutude, Inf=Largest Magnitude and any value inbetween)
# INPUTS (optional)
# M..................................................................................................................RHS matrix (typically a mass matrix, defaults to the identity matrix)
# maxiter..............................................................................................the maximum number of iterations (defaults to 100, only used for iterative methods)
# tol.............................................................................................the desired tolerance of the method (defaults to 1e-12, only used for iterative methods)
# v0...........................................................................an initial guess for the eigenvector, only applies to the high-precision methods (the last two in the file)
# OUTPUTS
# a named tuple with names:
# values................................................................................................................................the calculated eigenvalues (NB: this is a vector!)
# vectors.........................................................................................................................(a matrix whose columns are) the calculated eigenvectors
# error.........................................................................................the error with which the eigenvalues/vectors were calculated (always 0 for direct methods)
# converged...........................................................................a flag telling the user if the error is within the tolerance or not (always true for direct methods)
function HP_eigs(A, lambda; v0=rand(size(A,1)), M=eye(size(A,1)),  tol=BigFloat(1e-50), maxiter=1000)
    A=sparse(BigFloat.(A));
    M=sparse(BigFloat.(M));
    lambda=BigFloat(real(lambda))+1.0im*BigFloat(imag(lambda));
    P=ilu(A-M*lambda);
    iter=0;
    v=BigFloat.(v0)
    while iter<maxiter && norm(A * v - lambda * M * v)>tol
        v=gmres(A-M*lambda,v,Pl=P);
        v=v/norm(v);
        lambda=(v'*A*v)/(v'*M*v)
        iter=iter+1
    end
    err=norm(A * v - lambda * M * v)
    v=Float64.(real(v))+1.0im*Float64.(imag(v))
    lambda=Float64(real(lambda))+1.0im*Float64(imag(lambda))
    if err<tol
        conv=true
    else
        conv=false
    end
    return (vector=v, value=lambda, error=err, converged=conv)
end