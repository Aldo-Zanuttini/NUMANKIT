############# FUNCTION: mpinv #############
# computes the Moore-Penrose pseudoinverse
# of a matrix A making use of the built in
# sparse inverse algorithms in Julia and
# applies it to a vector x. If A is square
# then returns the normal inverse.
function mpinv(A,x)
    if size(A)[1]>size(A)[2]
        return (A'*A)\(A*x)
    elseif size(A)[1]<size(A)[2]
        return A'*((A*A')\x)
    else
        return A\x
    end
end
############## FUNCTION: eye ##############
# equivalent to matlab's eye() function,
# eye(n) returns an n-by-n sparse
# identity matrix.
function eye(n)
    return spdiagm(ones(n))
end
############ FUNCTION:  dirder ############
# numerical derivative of order ord of a
# function f, at a point x, in a direction
# v
function dirder(f,x,v,ord)
    h=10^(-16);
    if ord==1
        h=h^(1/2)*norm(x);
        df= (f(x+h*v)-f(x-h*v))/h;
    elseif ord==2
        h=h^(1/3)*norm(x);
        df= (f(x+h*v)-2*f(x)+f(x-h*v))/(h^2);
    elseif ord==3
        h=h^(1/4)*norm(x);
        df= (f(x+2*h*v)-2*f(x+h*v)+2*f(x-h*v)-f(x-2*h*v))/(2*h^3)
    elseif ord==4
        h=h^(1/5)*norm(x);
        df= (f(x+2*h*v)-4*f(x+h*v)+6*f(x)-4*f(x-h*v)+f(x-2*h*v))/(h^4);
    end
    return df
end

############# FUNCTION: newton #############
#    finds a root for the function f
# INPUTS
# x................initial guess for f(x)=0
# f................function to find the
#                  root of
# df...............jacobian of f
# tol..............tolerance of method
# maxit............maximum number of
#                  iterations
# OUTPUTS
# x................root of f
function newton(x,f,df,tol,maxit)
    iter=0;
    nrmdx=Inf;
    while nrmdx > tol && iter < maxit
        dx=-mpinv(df(x),f(x));
        x=x+dx;
        nrmdx=norm(dx);
        iter=iter+1;
    end
    if nrmdx < tol
        return (x=x, flag=string("converged in ", iter, " iterations"))
    else
        return (x=x, flag="not converged")
    end
end

########## FUNCTION: sparsematdiv ##########
# Computes the product
#              C=A^(-1)*B
# where A (and B) are sparse matrices
# INPUTS
# A...................sparse matrix whose
#                     inverse is to be
#                     applied
# B...................(sparse) matrix to
#                     be left-multiplied
#                     with a
# OUTPUTS
# result..............the product A^(-1)*B
function sparsematdiv(A,B)
    result=A\Vector(B[:,1])
    for i=2:size(B)[2]
        result=[result A\Vector(B[:,i])]
    end
    return dropzeros(sparse(result))
end

##### FUNCTION: truncated_eigendecomp  #####
# computes a factorization A=VDV' where
# size(A)>size(D)
# INPUTS
# A...................the matrix to be 
#                     factorised
# rank................the size of the matrix
#                     Df
# OUTPUTS
# V...................the matrix containing
#                     the eigenvectors
# D...................the matrix whose
#                     diagonal are the
#                     eigenvalues
function truncated_eigendecomp(M,rank)
    if size(M,1)==1
        V=M;
        D=1;
    elseif size(M[1,:])[1]>3
        D,V=eigs(A,nev=rank,which=:LM);
        D=spdiagm(D);
        V=dropzeros(sparse(V));
    else
        D,V=eigen(Matrix(M));
        D=spdiagm(D[end-rank+1:end]);
        V=V[:,end-rank+1:end];
    end
    return (V=V,D=D)
end




############ FUNCTION: savedata ############
# INPUTS
# data................the data to be saved
# name................the name of the file
#                     to be created
#                     e.g. "file.jl"
function savedata(data, name)
    @save name data
end

############ FUNCTION: loaddata ############
# INPUTS (compulsory)
# filename............the name of the file
#                     to be loaded
#                     e.g. "file.jl"
# INPUTS (optional)
# key.................the key of the saved
#                     data (string)
function loaddata(filename,key="data")
    data=load(filename)
    return data[key]
end