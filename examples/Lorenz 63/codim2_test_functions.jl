function get_lyapunov_coefficient(F,Fx,H;M=eye(length(H)-2))
    omega=abs(H[end]);
    lambda=H[end-1];
    X=H[1:end-2];
    A=Fx([X;lambda])[:,1:length(X)];
    _,q=jdqz(A,M,pairs=1,target=Near(1.0im*omega),tolerance=1e-12,max_iter=10^12,verbosity=1)
    _,p=jdqz(A',M',pairs=1,target=Near(-1.0im*omega),tolerance=1e-12,max_iter=10^12,verbosity=1)
    q=refine_eigenvector(A,q,1.0im*omega;mass_matrix=M);
    p=refine_eigenvector(A',p,-1.0im*omega;mass_matrix=M');
    q=q/norm(q);
    p=p/dot(p,q);
    B(u,v)=(dirder(x -> F([x;lambda]),X,u+v,2)-dirder(x -> F([x;lambda]),X,u-v,2))*0.25
    C(u,v,w)=(dirder(x -> F([x;lambda]),X,u+v+w,3)-dirder(x -> F([x;lambda]),X,u+v,3)-dirder(x -> F([x;lambda]),X,v+w,3)-dirder(x -> F([x;lambda]),X,u+w,3)+dirder(x -> F([x;lambda]),X,u,3)+dirder(x -> F([x;lambda]),X,v,3)+dirder(x -> F([x;lambda]),X,w,3))/48;
    h11=-A\B(q,conj(q));
    h20=(2.0im*omega*M-A)\B(q,q);
    l1=real(dot(p,C(q,q,conj(q))+2*B(q,h11)+B(conj(q),h20)))/(2*omega);
    return l1
end

function get_cubic_coefficient(F,Fx,LP;M=eye(length(LP)-1))
    lambda=LP[end];
    X=LP[1:end-1];
    A=Fx([X;lambda])[:,1:length(X)];
    p=get_critical_eigenvector(A';mass_matrix=M');
    q=get_critical_eigenvector(A;mass_matrix=M);
    q=q/norm(q);
    p=p/dot(p,q);
    B(u,v)=(dirder(x -> F([x;lambda]),X,u+v,2)-dirder(x -> F([x;lambda]),X,u-v,2))*0.25
    b=0.5*dot(p,B(q,q));
    return b
end
