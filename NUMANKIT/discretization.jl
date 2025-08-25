#################################################################################### FUNCTION: mkgrid ####################################################################################
# creates a staggered (and optionally stretched) grid
# INPUTS (compulsory)
# domain..................................................................................................................................................a Vector{Float64} with 2 entries
# number_of_points................................................................................................................the total number of points in the discretization (Int64)
# INPUTS (optional)
# stretching_factor................................................................................................................a factor by which the grid is to be stretched (Float64)
# region_of_interest................a string saying which area of the grid is supposed to have more points (possible values: "right boundary", "left boundary", "boundaries" and "center")
# OUTPUTS a named tuple with names:
# .points......................................................................................................................................a Vector{Float64} containing the gridpoints
# .volumes_plus.............................................................a Vector{Float64} containing the volumes (including the volume outside the boundary in the positive direciton)
# .volumes_minus............................................................a Vector{Float64} containing the volumes (including the volume outside the boundary in the negative direciton)
# .volumes_central.............................................................................................................a Vector{Float64} containing the volumes (excl. boundaries)
# .boundary_volume..............................................................................................the volume outside the domain (Float64, assumed to be equal on both sides)
function mkgrid(domain,number_of_points)
    if domain[2]>domain[1]
        x1=domain[1];
        x2=domain[2];
    else
        x1=domain[2];
        x2=domain[1];
    end
    dx=(x2-x1)/number_of_points;
    x1=x1+dx/2;
    x2=x2-dx/2;
    x=collect(range(x1,x2,number_of_points));
    return (points=x, volumes_minus=[dx; x[2:end]-x[1:end-1]], volumes_plus=[x[2:end]-x[1:end-1]; dx], volumes_center=x[2:end]-x[1:end-1], boundary_volume=dx)
end

function mkgrid(domain,number_of_points,stretching_factor,region_of_interest)
    dx=mkgrid(domain,number_of_points).boundary_volume;
    X=mkgrid(domain,number_of_points).points;
    x1=X[1];
    x2=X[end];
    a=abs(stretching_factor)
    if region_of_interest=="right boundary"
        x=((x1-x2).*exp.(-a.*X) .+ exp(-a*x1)*x2 .- exp(-a*x2)*x1)./(exp(-a*x1)-exp(-a*x2));
    elseif region_of_interest=="left boundary"
        x=((x1-x2).*exp.(a.*X) .+ exp(a*x1)*x2 .- exp(a*x2)*x1)./(exp(a*x1)-exp(a*x2));
    elseif region_of_interest=="boundaries"
        x=0.5*(x2-x1)*tanh.(a*(X.-(x1+x2)/2))/tanh(a*(x2-x1)/2) .+ (x1+x2)/2;
    elseif region_of_interest=="center"
        x=atanh.(tanh(a*(x2-x1)/2)*(2*X .- (x1+x2))./(x2-x1))./a .+ (x1+x2)/2;
    end
    return (points=x, volumes_minus=[dx; x[2:end]-x[1:end-1]], volumes_plus=[x[2:end]-x[1:end-1]; dx], volumes_center=x[2:end]-x[1:end-1], boundary_volume=dx)
end


################################################################################## FUNCTION: discretize ##################################################################################
#                                                             creates a discrete version of some differential operator
#                                                                 (currently only allows diffusion and convection)
# INPUTS (compulsory)
# operator............................................................................................a string describing what operator is to be discretized ("convection" or "diffusion")
# grid.......................................................................................................................................a named tuple produced by the mkgrid function
# boundary_conditions..............a named tuple (plus="neumann/dirichlet/robin", minus="neumann/dirichlet/robin") describing what type of homogeneous boundary condition is to be used at
#                                  each end of the domain. When choosing robin conditions, the form
#                                                                                       u+b*u_x=0
#                                  is assumed, in which case the named tuple must also contain the name
#                                                                                        coeff=b.
#                                  If a name is not in the set {dirichlet, neumann, robin} then no boundary conditions will be applied on that end of the domain.
# INPUTS (optional)
# grid1, grid2......................................................................................................................grids for the other physical dimensions of the problem
# boundary_conditions1, boundary_conditions2...............................................................................boundary conditions for other physical dimension of the problem
# OUTPUTS (1D)
# when asked for diffusion a matrix corresponding to the discrete version of the diffusion, when asked
# for convection, a tuple with names:
# .convection_plus........................................................................................................................................convection as (u_{n+1}-u_{n})/dx
# .convection_minus.......................................................................................................................................convection as (u_{n}-u_{n-1})/dx
# OUTPUTS (2D and 3D) when discretising diffusion
# .laplacian...................................................................................................................................sum of the x, y (and z) direction diffusion
# .x_diffusion............................................................................................................................................matrix corresponding to d^2/dx^2
# .y_diffusion............................................................................................................................................matrix corresponding to d^2/dy^2
# .z_diffusion............................................................................................................................................matrix corresponding to d^2/dz^2
# OUTPUTS (2D and 3D) when discretising convection
# .x_convection_plus............................................................................................................................matrix corresponding to (u_{n+1}-u_{n})/dx
# .x_convection_minus...........................................................................................................................matrix corresponding to (u_{n}-u_{n-1})/dx
# .y_convection_plus............................................................................................................................matrix corresponding to (u_{n+1}-u_{n})/dy
# .y_convection_minus...........................................................................................................................matrix corresponding to (u_{n}-u_{n-1})/dy
# .z_convection_plus............................................................................................................................matrix corresponding to (u_{n+1}-u_{n})/dz
# .z_convection_minus...........................................................................................................................matrix corresponding to (u_{n}-u_{n-1})/dz
function discretize(operator,grid,boundary_conditions)
    if operator=="convection"
        U=1 ./grid.volumes_center;
        L=-1 ./grid.volumes_center;
        Dplus=-1 ./grid.volumes_plus;
        Dminus=1 ./grid.volumes_minus;
        if boundary_conditions.plus=="neumann"
            Dplus[end]+=-Dplus[end];
        elseif boundary_conditions.plus=="dirichlet"
            Dplus[end]+=Dplus[end];
        elseif boundary_conditions.plus=="robin"
            Dplus[end]+=((boundary_conditions.coeff/grid.boundary_volume+0.5)/(boundary_conditions.coeff/grid.boundary_volume-0.5))/grid.boundary_volume;
        end
        if boundary_conditions.minus=="neumann"
            Dminus[1]+=-Dminus[1];
        elseif boundary_conditions.minus=="dirichlet"
            Dminus[1]+=Dminus[1];
        elseif boundary_conditions.minus=="robin"
            Dminus[1]+=-((boundary_conditions.west_coeff/grid.boundary_volume+0.5)/(boundary_conditions.coeff/grid.boundary_volume-0.5))/grid.boundary_volume;
        end
        return (convection_plus= spdiagm(0 => Dplus, 1 => U), convection_minus= spdiagm(-1 => L, 0 => Dminus))
    elseif operator=="diffusion"
        components=discretize("convection",grid,boundary_conditions);
        return components.convection_plus-components.convection_minus
    end
end

function discretize(operator, grid, grid1, boundary_conditions, boundary_conditions1)
    Ixdx=spdiagm(ones(length(grid.points))).*(grid.volumes_plus+grid.volumes_minus)/2;
    Iydy=spdiagm(ones(length(grid1.points))).*(grid1.volumes_plus+grid1.volumes_minus)/2;
    if operator=="diffusion"
        Dxx=discretize(operator, grid, boundary_conditions);
        Dyy=discretize(operator, grid1, boundary_conditions1);
        Dxx=kron(Iydy,Dxx);
        Dyy=kron(Dyy,Ixdx);
        return (laplacian=Dxx+Dyy, x_diffusion=Dxx, y_diffusion=Dyy)
    elseif operator=="convection"
        Dx_plus, Dx_minus=discretize(operator, grid, boundary_conditions);
        Dy_plus, Dy_minus=discretize(operator, grid1, boundary_conditions1);
        Dx_plus=kron(Iydy,Dx_plus);
        Dx_minus=kron(Iydy,Dx_minus);
        Dy_plus=kron(Dy_plus,Ixdx);
        Dy_minus=kron(Dy_minus,Ixdx);
        return (x_convection_plus=Dx_plus, x_convection_minus=Dx_minus, y_convection_plus=Dy_plus, y_convection_minus=Dy_minus)
    end
end

function discretize(operator, grid, grid1, grid2, boundary_conditions, boundary_conditions1, boundary_conditions2)
    Ixdx=spdiagm(ones(length(grid.points))).*(grid.volumes_plus+grid.volumes_minus)/2;
    Iydy=spdiagm(ones(length(grid1.points))).*(grid1.volumes_plus+grid1.volumes_minus)/2;
    Izdz=spdiagm(ones(length(grid2.points))).*(grid2.volumes_plus+grid2.volumes_minus)/2;
    if operator=="diffusion"
        Dzz=discretize(operator, grid2, boundary_conditions2);
        laplacian, x_diffusion, y_diffusion=discretize(operator,grid,grid1,boundary_conditions,boundary_conditions1)
        Dxx=kron(x_diffusion,Izdz);
        Dyy=kron(y_diffusion,Izdz);
        Dzz=kron(kron(Ixdx,Iydy),Dzz);
        return (laplacian=Dxx+Dyy+Dzz, x_diffusion=Dxx, y_diffusion=Dyy, z_diffusion=Dzz)
    elseif operator=="convection"
        Dz_plus, Dz_minus=discretize(operator, grid2, boundary_conditions2);
        x_convection_plus, x_convection_minus, y_convection_plus, y_convection_minus=discretize(operator,grid,grid1,boundary_conditions,boundary_conditions1);
        Dx_plus=kron(x_convection_plus,Izdz);
        Dx_minus=kron(x_convection_minus,Izdz);
        Dy_plus=kron(y_convection_plus,Izdz);
        Dy_minus=kron(y_convection_minus,Izdz);
        Dz_plus=kron(kron(Ixdx,Iydy),Dz_plus);
        Dz_minus=kron(kron(Ixdx,Iydy),Dz_minus)
        return (x_convection_plus=Dx_plus, x_convection_minus=Dx_minus, y_convection_plus=Dy_plus, y_convection_minus=Dy_minus, z_convection_plus=Dz_plus, z_convection_minus=Dz_minus)
    end
end
################################################################################### FUNCTION: meshgrid ###################################################################################
#                     adapts 1D gridpoints vectors to 2D or 3D gridpoints vectors
# INPUTS (compulsory)
# x.............................................................................................................................................vector of 1D gridpoints in the x direction
# y.............................................................................................................................................vector of 1D gridpoints in the y direction
# INPUTS (optional)
# z.............................................................................................................................................vector of 1D gridpoints in the z direction
# OUTPUTS (2D) a named tuple with names
# .X.................................................................................................................................................................2D x gridpoint vector
# .Y.................................................................................................................................................................2D y gripdoint vector
# OUTPUTS (3D) a named tuple with names
# .X.................................................................................................................................................................3D x gridpoint vector
# .Y.................................................................................................................................................................3D y gripdoint vector
# .Z.................................................................................................................................................................3D z gripdoint vector
function meshgrid(x,y)
    x=kron(ones(length(y)),x);
    y=kron(ones(length(x)),y);
    return (X=x,Y=y)
end
function meshgrid(x,y,z)
    x,y=meshgrid(x,y);
    x=kron(x,ones(length(z)));
    y=kron(x,ones(length(z)));
    z=kron(kron(ones(length(x)),ones(length(z))),z);
    return (X=x,Y=y,Z=z)
end