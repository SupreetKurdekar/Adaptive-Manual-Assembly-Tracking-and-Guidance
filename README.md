# Adaptive Manual Assembly Tracking and Guidance

## Introduction

  The idea behind this project is to create an interactive system between a human and a computer that efficiently tracks the assembly process without hindering the workflow of the human operator. This simple tracking system can save many hours of laborious video annotation that is traditionally required in the process of assembly sequence optimization and design for assembly prototyping. 

  While such systems already exist most of them are stuck to one rigid predefined assembly sequence. This system allows one to explore different assembly sequences, to select bad sequences, remembering them for future users and also affords the user the opportunity to return to a feasible state in the assembly.   

## Connectivity Graph
The assembly component connectivity graph is the only input required by the system. In a commercial setting this can be directly extracted from a CAD model of the assembly.

![chair_connect](https://github.com/SupreetKurdekar/Adaptive-Manual-Assembly-Tracking-and-Guidance/blob/main/docs/images/chair_connect2.PNG)
## AND/OR Tree for Assembly Sequence Encoding

This AND/OR Tree is the heart of the system. The tree captures all possible assembly sequences - feasible and infeasible. This tree is automatically generated from the Connectivity Graph.

Each node represents a sub-assembly among all possible sub-assemblies in final subassembly.

The hierarchy shows the direction of assembly, wherein addition of each new piece, takes the system one assembly step further.

**Geometrical Infeasibility** is when two sub-assemblies have no connectivity in the connectivity according to the connectivity graph.

**Mechanical Infeasibility** is when a previous connection makes further connections impossible.

![Chair_and_or](https://github.com/SupreetKurdekar/Adaptive-Manual-Assembly-Tracking-and-Guidance/blob/main/docs/images/Chair_and_or.PNG)

## Experimental Setup for assembly of Hand Drill

![Experimental_Setup](https://github.com/SupreetKurdekar/Adaptive-Manual-Assembly-Tracking-and-Guidance/blob/main/docs/images/env_set_1_1.png)
![Experimental_Setup](https://github.com/SupreetKurdekar/Adaptive-Manual-Assembly-Tracking-and-Guidance/blob/main/docs/images/env_set_2.png)


## Initial AND/OR Tree after using Connectivity Information

![image](https://user-images.githubusercontent.com/49178343/125210698-765d3800-e26f-11eb-9c4f-1c02d5451bd2.png)


## Demo

Yellow Boxes depict which objects the system thinks can be assembled next. Over time, the system understands which ones are truly feasible.
We can see as more infeasible sequences are found, better suggestions are depicted.

https://user-images.githubusercontent.com/49178343/125202866-a858a500-e243-11eb-982e-b0dd663c788b.mp4

## Updated AND/OR Tree after single experiment

[image](https://user-images.githubusercontent.com/49178343/125210540-609b4300-e26e-11eb-9320-84a7c5161421.png)


## Hueristics to find Feasible Sequences

A heuristic is any approach to problem solving or self-discovery that employs a practical method that is not guaranteed to be optimal but is nevertheless sufficient for reaching an immediate, short-term goal or approximation.

The system allows an expert to encode such heuristics into the graph. 
We find the set of nodes that specify a certain user-defined critera as infeasible.
The system automatically sets the correct edges to be infeasible.

### Heuristics for Drill Assembly

1) The left and right handles must be attached only after all internal parts have been connected.
2) The battery must be assembled in the very end.

## Final AND/OR Tree after applying Heuristics

![Feasible_drill_tree](https://github.com/SupreetKurdekar/Adaptive-Manual-Assembly-Tracking-and-Guidance/blob/main/docs/images/Feasible_drill_tree.PNG)

## TODO

1) Extract final feasible graph.
2) Apply dynamic programming approach to finding the shortest time sequence to time weighted tree

## Hardware requirements

1) Realsense RGBD camera. 

## Software requirements

Libraries required:

1) PyRealsense
2) networkx
3) OpenCv
4) Plotly
5) Matplotlib
6) Numpy
7) Sci-Py
8) Pickle

