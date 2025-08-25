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
    nrmx=norm(x);
    if nrmx==0
        nrmx=1;
    end
    if ord==1
        h=h^(1/2)*nrmx;
        df= (-f(x+3*h*v)+9*f(x+2*h*v)-45*f(x+h*v)+45*f(x-h*v)-9*f(x-2*h*v)+f(x-3*h*v))/(60*h);
    elseif ord==2
        h=h^(1/3)*nrmx;
        df= (2*f(x+3*h*v)-27*f(x+2*h*v)+270*f(x+h*v)-490*f(x)+270*f(x-h*v)-27*f(x-2*h*v)+2*f(x-3*h*v))/(180*h^2);
    elseif ord==3
        h=h^(1/4)*nrmx;
        df= (-f(x+3*h*v)+8*f(x+2*h*v)-13*f(x+h*v)+13*f(x-h*v)-8*f(x-2*h*v)+f(x-3*h*v))/(8*h^3);
    elseif ord==4
        h=h^(1/5)*nrmx;
        df= (9*f(x+4*h*v)-128*f(x+3*h*v)+1008*f(x+2*h*v)-8064*f(x+h*v)+14350*f(x)-8064*f(x-h*v)+1008*f(x-2*h*v)-128*f(x-3*h*v)+9*f(x-4*h*v))/(5040*h^4);
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

########## FUNCTION: matdiv ##########
# Computes the product
#              c=A^(-1)*b
# where A (and B) are sparse matrices
# INPUTS
# A...................sparse matrix whose
#                     inverse is to be
#                     applied
# b...................vector to be left
#                     multiplied
#                     with A inverse
# tolerance...........the accuracy of
#                     the product (optional)
# OUTPUTS
# result..............the product A^(-1)*b
function matdiv(A,b;tolerance=1e-16)
A=sparse(A)
answer=nothing
    if length(b)<10^3 || tolerance==0
        try
            answer=A\b
        catch e
            if isa(e,SingularException)
                epsilon=1e-6*max(length(b),opnorm(A,Inf))
                P=ilu(A+eye(size(A,1))*epsilon,τ=0)
                answer=P\b
            else
                try
                    P=ilu(A,τ=0)
                    answer=P\b
                catch e
                 throw(e)
                end
            end
        end
    elseif length(b)>=10^3
        try
            answer=gmres(A,b,abstol=tolerance,reltol=tolerance)
        catch e
            if isa(e,LinearAlgebra.LAPACKException)
                P=ilu(A)
                answer=gmres(A,b,Pl=P,abstol=tolerance,reltol=tolerance)
            end
        end
    end
    return answer
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
    else
        D,V,_=eigz(M,Inf,howmany=rank,tol=1e-15);
        D=spdiagm(D);
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