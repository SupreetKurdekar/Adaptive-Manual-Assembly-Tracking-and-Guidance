# Adaptive Manual Assembly Tracking and Guidance

## Introduction

  The idea behind this project is to create an interactive system between a human and a computer that efficiently tracks the assembly process without hindering the workflow of the human operator. This simple tracking system can save many hours of laborious video annotation that is traditionally required in the process of assembly sequence optimization and design for assembly prototyping. 

  While such systems already exist most of them are stuck to one rigid predefined assembly sequence. This system allows one to explore different assembly sequences, to select bad sequences, remembering them for future users and also affords the user the opportunity to return to a feasible state in the assembly.   

## Connectivity Graph
The assembly component connectivity graph is the only input required by the system. In a commercial setting this can be directly extracted from a CAD model of the assembly.

![chair_connect](docs/images/chair_connect2.png)
## AND/OR Tree for Assembly Sequence Encoding

This AND/OR Tree is the heart of the system. The tree captures all possible assembly sequences - feasible and infeasible. This tree is automatically generated from the Connectivity Graph.

Each node represents a sub-assembly among all possible sub-assemblies in final subassembly.

The hierarchy shows the direction of assembly, wherein addition of each new piece, takes the system one assembly step further.

**Geometrical Infeasibility** is when two sub-assemblies have no connectivity in the connectivity according to the connectivity graph.

**Mechanical Infeasibility** is when a previous connection makes further connections impossible.

![Chair_and_or](docs/images/Chair_and_or.png)

##
