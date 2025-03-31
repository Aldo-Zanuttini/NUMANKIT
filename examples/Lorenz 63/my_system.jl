# Define a simple dynamical system
f(x,alpha,sigma,beta)=[sigma*(x[2]-x[1]);x[1]*(alpha-x[3])-x[2];x[1]*x[2]-beta*x[3]];
# Define the jacobian
Df(x,alpha,sigma,beta)=[-sigma sigma 0; alpha-x[3] -1 -x[1]; x[2] x[1] -beta];

# Specify some parameter values
sigma=10;
beta=8/3;

# Define the augmented system for continuation
F(x)=f(x[1:end-1],x[end],sigma,beta); # augmented RHS
df_dalpha(x)=[0; x[1]; 0]; # derivative of the RHS wrt the continuation parameter
DF(x)=[Df(x[1:end-1],x[end],sigma,beta) df_dalpha(x)] # augmented jacobian