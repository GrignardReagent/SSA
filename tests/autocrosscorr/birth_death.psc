# Stochastic Simulation Algorithm input file


# Reactions
R1:
    $pool > A
    k

R2:
    A > $pool
    d*A

# Fixed species

# Variable species
A= 0

# Parameters
k= 0.1
d= 0.01
