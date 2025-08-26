
########################################################################## FUNCTION: min_aug_system_zero_eigval ##########################################################################
#                                                                                 Given a system of the form
#                                                                      Mu_t=f(u,lambda),  Df(u,lambda)=df(u,lambda)/du     (1)
#                                                           and its associated augmented form used for continuation of equilibria
#                                                            Mu_t=F(x),   Fx(x)=[Df(u,lambda), df(u,lambda)/dlambda],   x=(u,lambda)    (2)
#                                                                this function builds a new augmented system from (2) of the form
#                                                                                      G(x)=[F(x);g(x)]        (3)
#                                                                               where g(x) is the solution of
#                                             [Df(u,lambda), p; 0 q']*[w(u,lambda);g(u,lambda)]=[0;1], where Df(u,lambda)*q=Df(u,lambda)'p=0
#                         this system is nonsingular when Df(u,lambda) is singular. The jacobian is computed in the associated function min_aug_jacobian_zero_eigval.
#                                                      This function is used mainly as support for the fold/branchpoint locating functions.
# INPUTS
# x.................................................................................................................a point suitable for system (2) at which system (3) is to be evaluated
# F..............................................................................................................................................function handle for the RHS of system (2)
# Fx.............................................................................................................................function handle for the jacobian of the RHS of system (2)
# M........................................................................................................................the mass matrix of system (1) (defaults to the identity matrix)
# OUTPUTS
# G.................................................................................................................................the value of the function of system (3) at the point x
function min_aug_system_zero_eigval(x,F,Fx;M=eye(length(x)-1))
    Df=Fx(x)[:,1:end-1];
    n=size(Df,1);
    q=eigz(Df,0;M=M)[2];
    p=eigz(Df',0;M=M')[2];
    p=p/(p'*q);
    g=[Df p;q' 0]\[zeros(n);1];
    g=g[end];
    G=[F(x);g]
    return G
end

######################################################################### FUNCTION: min_aug_jacobian_zero_eigval #########################################################################
#                                                                                 Given a system of the form
#                                                                      Mu_t=f(u,lambda),  Df(u,lambda)=df(u,lambda)/du     (1)
#                                                           and its associated augmented form used for continuation of equilibria
#                                                            Mu_t=F(x),   Fx(x)=[Df(u,lambda), df(u,lambda)/dlambda],   x=(u,lambda)    (2)
#                                                                this function builds the jacobian of the new augmented system
#                                                                                        G(x)=[F(x);g(x)]                      (3)
#                                                                 (for more info on this system see: codim2_min_aug_system(...))
#                                                                                 this jacobian has the form
#                                                                                    DG(x)=[Fx(x);dg/dx]
#                                                      where dg/dx=-p'*(dFx/dx)*q and (p,q) are such that Df(u,lambda)*q=Df(u,lambda)'*p
#                                                     This function is used mainly as support for the bifurcation locating functions below.
# INPUTS
# x.................................................................................................................a point suitable for system (2) at which system (3) is to be evaluated
# F..............................................................................................................................................function handle for the RHS of system (2)
# Fx.............................................................................................................................function handle for the jacobian of the RHS of system (2)
# M........................................................................................................................the mass matrix of system (1) (defaults to the identity matrix)
# OUTPUTS
# DG................................................................................................................................the value of the jacobian of system (3) at the point x
function min_aug_jacobian_zero_eigval(x,F,Fx;M=eye(length(x)-1))
    Df=Fx(x)[:,1:end-1];
    n=size(Df,1);
    q=eigz(Df,0;M=M)[2];
    p=eigz(Df',0;M=M')[2];
    q=q/(p'*q);
    wg=[Df p;q' 0]\[zeros(n);1];
    w=wg[1:end-1];
    g=wg[end];
    vh=[Df' q;p' 0]\[zeros(n);1];
    v=vh[1:end-1];
    Bq=(Fx(x+10^(-8)*norm(x)*[w;0])-Fx(x-10^(-8)*norm(x)*[w;0]))/(2*10^(-8)*norm(x));
    gprime=-v'*Bq;
    DG=[Fx(x);gprime]
    return DG
end

############################################################################# FUNCTION:  min_aug_system_hopf #############################################################################
#                                                                                 Given a system of the form
#                                                                      Mu_t=f(u,lambda),  Df(u,lambda)=df(u,lambda)/du     (1)
#                                                           and its associated augmented form used for continuation of equilibria
#                                                            Mu_t=F(x),   Fx(x)=[Df(u,lambda), df(u,lambda)/dlambda],   x=(u,lambda)    (2)
#                                                                this function builds a new augmented system from (2) of the form
#                                                                                  G(x)=[F(x);g_1(x);g_2(x)]        (3)
#                                                        where g_1(x) and g_2(x) are the real (resp. imaginary) parts of the solution of
#                          [Df(u,lambda)-I*omega, p; 0 q']*[w(u,lambda, omega);g(u,lambda, omega)]=[0;1], where (Df(u,lambda)-I*omega)*q=(Df(u,lambda)-I*omega)'p=0
#                                                    and omega is the imaginary component of the eigenvalue corresponding to the hopf bifurcation.
#                         this system is nonsingular when Df(u,lambda)-I*omega is singular. The jacobian is computed in the associated function min_aug_jacobian_hopf.
#                                                           This function is used mainly as support for the hopf locating function.
# INPUTS
# x......................................................................................................................a point x=(u,lambda,omega) at which system (3) is to be evaluated
# F..............................................................................................................................................function handle for the RHS of system (2)
# Fx.............................................................................................................................function handle for the jacobian of the RHS of system (2)
# M........................................................................................................................the mass matrix of system (1) (defaults to the identity matrix)
# OUTPUTS
# G.................................................................................................................................the value of the function of system (3) at the point x
function min_aug_system_hopf(x,F,Fx;M=eye(length(x)-2))
    omega=x[end];
    x=x[1:end-1];
    Df=Fx(x)[:,1:end-1]
    n=size(Df,1);
    q=eigz(Df,1.0im*omega;M=M)[2];
    p=eigz(Df',-1.0im*omega;M=M')[2];
    q=q/norm(q);
    p=p/(p'*q);
    g=[Df p;q' 0]\[zeros(n);1];
    g=g[end];
    g1=real(g);
    g2=imag(g);
    G=[F(x);g1;g2]
    return G
end

############################################################################ FUNCTION:  min_aug_jacobian_hopf ############################################################################
#                                                                                 Given a system of the form
#                                                                      Mu_t=f(u,lambda),  Df(u,lambda)=df(u,lambda)/du     (1)
#                                                           and its associated augmented form used for continuation of equilibria
#                                                            Mu_t=F(x),   Fx(x)=[Df(u,lambda), df(u,lambda)/dlambda],   x=(u,lambda)    (2)
#                                                                this function builds the jacobian of the new augmented system
#                                                                                   G(x)=[F(x);g_1(x);g_2(x)]                      (3)
#                                                             (for more info on this system see: codim2_min_aug_system_hopf(...))
#                                                                                 this jacobian has the form
#                                                                                    DG(x)=[Fx(x);dg/dx]
#                                              where dg/dx=-p'*(d(Fx-[I*i*omega 0])/dx)*q and (p,q) are such that Df(u,lambda)*q=Df(u,lambda)'*p
#                                                        This function is used mainly as support for the bifurcation locating functions.
# INPUTS
# x......................................................................................................................a point x=(u,lambda,omega) at which system (3) is to be evaluated
# F..............................................................................................................................................function handle for the RHS of system (2)
# Fx.............................................................................................................................function handle for the jacobian of the RHS of system (2)
# M........................................................................................................................the mass matrix of system (1) (defaults to the identity matrix)
# OUTPUTS
# DG................................................................................................................................the value of the jacobian of system (3) at the point x
function min_aug_jacobian_hopf(x,F,Fx;M=eye(length(x)-2))
    omega=x[end];
    x=x[1:end-1];
    Df=Fx(x)[:,1:end-1];
    n=size(Df,1);
    q=eigz(Df,1.0im*omega;M=M)[2];
    p=eigz(Df',-1.0im*omega,M=M')[2];
    q=q/norm(q);
    p=p/(p'*q);
    wg=[Df p;q' 0]\[zeros(n);1];
    w=wg[1:end-1];
    vh=[Df' q;p' 0]\[zeros(n);1];
    v=vh[1:end-1];
    Bq=(Fx(x+10^(-8)*norm(x)*[w;0])-Fx(x-10^(-8)*norm(x)*[w;0]))/(2*10^(-8)*norm(x));
    gprime=[-v'*Bq v'*M*w*1.0im];
    gprime1=real(gprime);
    gprime2=imag(gprime);
    DG=[Fx(x) zeros(n);gprime1;gprime2]
    return DG
end