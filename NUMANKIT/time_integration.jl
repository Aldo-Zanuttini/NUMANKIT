################################################################################## FUNCTION: timeseries ##################################################################################
#                                                           creates a time-series for some (non) autonomous dynamical system
#                                                                                        Mx'=f(t,x)
#                                             also accepts autonomous systems and non-DAE systems (note that if M is singular one must
#                                                choose an implicit method (theta below can be chosen to be very small... note also
#                                                  if M is not the identity matrix, i.e. the timescales are meaningfully different 
#                                                                        one must choose a suitably small timestep)
# INPUTS (compulsory)
# x0.....................................................................................................................................................initial condition for the problem
# steps............................................................................................................................................................number of steps to make
# stepsize..........................................................................................................................................................the value of direction
# theta..........................the parameter deciding what method to use: for theta=0 returns backward Euler, for theta=1 returns forward Euler and for theta=0.5 returns Crank-Nicolson
# f.........................................................................................................................................................the RHS function of Mx'=f(t,x)
# M..........................................................................................................................................................the mass matrix of Mx'=f(t,x)
# df..............................................................................................................................................................the Jacobian matrix of f
# INPUTS (optional)
# t0...................................................................................................................................initial time, necessary if system is non-autonomous
# newton_tol..................................................................................................................................................tolerance of newton's method
# newton_maxit.........................................................................................................................................maximum number of newton iterations
# M........................................................................................................................................................the mass matrix M in Mx'=f(t,x)
# OUTPUTS a named tuple with names
# .series...................................................a matrix whose column space are points in the time series (and whose rows are values of the variables in the dynamical system)
# .time..............................................................................................................a vector containing the subset of the time-set covered by integration
function timeseries(x0,t0,steps,stepsize,theta,f,df;newton_tol=10^(-3)*stepsize,newton_maxiter=10^3,M=1.0)
    if methods(f)[1].nargs==2
        x, T=timeseries(x0,steps,stepsize,theta,f,df,newton_tol,newton_maxiter)
        return (series=x, time=T.+t0)
    else
        x=zeros(length(x0),steps);
        T=zeros(steps);
        T[1]=t0;
        x[:,1]=x0;
        for i=2:steps
            yold=x[:,i-1];
            g(ynew)=M*ynew-stepsize*((1-theta)*f(t0+i*stepsize,ynew)+theta*f(t0+(i-1)*stepsize,yold))-M*yold;
            Dg(y)=M*eye(length(x0))-(1-theta)*stepsize*df(t0+i*stepsize,y);
            x[:,i]=newton(x[:,i-1],g,Dg,newton_tol,newton_maxiter).x
            T[i]=t0+i*stepsize;
        end
        return (series=x, time=T)
    end
end

function timeseries(x0,steps,stepsize,theta,f,df;newton_tol=10^(-3)*stepsize,newton_maxiter=10^3,M=1.0)
    if methods(f)[1].nargs==3
        return timeseries(x0,0,steps,stepsize,theta,f,df,newton_tol,newton_maxiter)
    else
        x=zeros(length(x0),steps);
        T=zeros(steps);
        x[:,1]=x0;
        for i=2:steps
            yold=x[:,i-1];
            g(ynew)=M*ynew-stepsize*((1-theta)*f(ynew)+theta*f(yold))-M*yold;
            Dg(y)=M*eye(length(x0))-(1-theta)*stepsize*df(y);
            x[:,i]=newton(x[:,i-1],g,Dg,newton_tol,newton_maxiter).x
            T[i]=(i-1)*stepsize;
        end
        return (series=x, time=T)
    end
end
