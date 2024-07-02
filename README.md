# Ising-on-SAW-2D

This program runs a Monte Carlo simulation of an Ising model on top of a self-avoiding random walk (SAW). 
More specifically, the Metropolis-Hasting algorithm is used. 

The variables obtained during these simulations are written to a file, such that we can use this data to obtain figures and histograms.
Along these variables is also the adjacency graphs of specific snapshots during the simulation. These adjacency graphs can be used to 
calculate the Ollivier-Ricci curvature in a mathematica program (still to be added). 

# Configuration for the SAW

The program is capable of initializing different configurations for the Ising model on the SAW. These different configurations are 
easy to change within the code itself and several possoibilities are already present. 
