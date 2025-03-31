Go through the files in this example in the following order:
0. import_NumAnKit.jl - this file is essential to import all the kit's functions
1. my_system.jl - this file tells you how to define your systems (this may be trivial for small systems such as Lorenz 63...)
2. first_equilibrium_branch.jl - this file should inform you on how to do continuation and use the centralized "analyse_branch()" function
3. second_equilibrium_branch.jl - this file should inform you on how to switch branches in case of a branchpoint
4. stability_of_cycle.jl - this file should inform you on how to use the "time_series()" function and on some of the names returned by the "analyse_branch()" function
