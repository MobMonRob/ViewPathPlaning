# Model based ViewPathPlanning
This repository contains code to fully generate viewpoints and robot positions to completely scan the surface of a given object. The implementation uses a 6 DOF robotic arm (UR5e) mounted with a 3D Scanner (micro epsilon suface control 3D 3510-80). The aproach is divided into 5 phases, which result in a list of robot position coordinates (in voxel unit) and corresponding joint configurations, both including a sequence indicator.
## PHASE 1
### Description
The first phase resulst in a list of viewpoint (VP) candidates and their corresponing visible object surface (represented as voxels). The phase includes the following steps:
- Initilization: load object model (e.g. mesh)
- Step 1: Voxelize object model as ocupancy matrix. Extend the matrix as preparation for the dilation step.
- Step 2: Generate viepoint sample space via binary dilation. Select random viewpoint candidates from the sample space. (10% recommended)
- Step 3: Calculate object voxels that are within the "potential field" of the viewpoints. Use the voxels of the potential field to calculate mean viewing direction of the viepoints
- Step 4: Calculate object voxels that are within the measurement area  (MA) of the viewpoints. Only obj voxels in the potential field are used per VP in terms of dimensionality reduction.
- Step 5: Ray tracing for ocllusion culling per object voxels in MA per VP. Only object voxels that are not occluded form the viepoits perspective are included to coount as "visible surface"
### Input and configuration
