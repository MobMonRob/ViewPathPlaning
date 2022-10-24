# Model based ViewPathPlanning
This repository contains code to fully generate viewpoints and robot positions to completely scan the surface of a given object. The implementation uses a 6 DOF robotic arm (UR5e) mounted with a 3D Scanner (micro epsilon suface control 3D 3510-80). The aproach is divided into 5 phases, which result in a list of robot position coordinates (in voxel unit) and corresponding joint configurations, both including a sequence indicator. Output file format is JSON.
## PHASE 1
### Description
The first phase resulst in a list of viewpoint (VP) candidates and their corresponing visible object surface (represented as voxels). The phase includes the following steps:
- Initialization: load object model (e.g. mesh)
- Step 1: Voxelize object model as ocupancy matrix. Extend the matrix as preparation for the dilation step.
- Step 2: Generate viepoint sample space via binary dilation. Select random viewpoint candidates from the sample space. (10% recommended)
- Step 3: Calculate object voxels that are within the "potential field" of the viewpoints. Use the voxels of the potential field to calculate mean viewing direction of the viepoints
- Step 4: Calculate object voxels that are within the measurement area  (MA) of the viewpoints. Only obj voxels in the potential field are used per VP in terms of dimensionality reduction.
- Step 5: Ray tracing for ocllusion culling per object voxels in MA per VP. Only object voxels that are not occluded form the viepoits perspective are included to coount as "visible surface"
### Input and configuration
- gridSize (this is the voxel size used)
- obj_mesh_scale (this is for scaling the object model before processing)
- number_of_vp_candidates_seed (if 1 or smaller than 1, it is used as percentage, else it is used as absolute number)
- sampleSpace_extension (this controls how far the smaple sae will be extended bejond the used max sensor range)
- ma_scale (scale the actual measurement area of the scanner, scaling smaller means there will be more overlap of the actual measurement areas of the VPs)
- sensor attributes (scanner dimensions, measurment areas, measurement ranges)
- obj_mesh_path (path to object model file)
- outputPath (located at the end of the code, there the output will be saved)

## PHASE 2
### Description
The second phase uses the generated VP candidates and their corresponding visible object voxels as set covering problem (SCP). The least amount of nessesary VPs need to be found that still cover the whole object surface. There are implemented the following algorithms:
- Greedy algorithm
- Naive algorithm
- Simulated annealing 
- Particle swarm optimization (with and without greedy input)
- Genetic algorithm (with and without greedy input)
- Chemical reaction optimization (with and without greedy input)

### Input and configuration
- inputPath (path to a phase 1 output file)
- algorithms (need to be commented out, if they should not be applied. Algorithm Paramters are set in file SCP_src)
- outputPath (located at the end of the code, there the output will be saved)


## PHASE 3
### Description
The third phase results in a list of robot position candidates and their corresponding reachable VPs. The phase includes the following steps:
- Step 1: Generate robot position sample space via binary dilation. Select random robot position candidates from the sample space. (10% recommended)
- Step 2: Get robot positioning orientation. The robot should always be oriented pointing away from the object with its optical axis. Thus the scanner is alway viewing away from the object in start position. This is for accessiblity purposes of the scanner while setting up the new position and it prevents invalid positions e.g. with the robots base beside the object but its rotated in a manner that the arm is under the object.
- Step 3: Calculate VP voxels that are within the "potential field" of the robots position. This is bassed on the working radius of the robot. As there could be VPs oriented in a manner that they are nevertheless reachable, this set is updated in the following step.
- Step 4: Get the robots joint configurations to reach the assigned VPs. Only VPs in the potential field are used in terms of dimensionality reduction. VPs that are not reachable will be deleted form the assigned VPS of the position. The inverse kinematics always result in 8 configurations possible. A selection of the most suitable configuration is not addressed. There will be always taken the first out of the 8 configurations.

### Input and configuration
- inputPath (path to a phase 2 output file)
- tm_rb_minRange (controls the placement of the robet, minimum distance to the object)
- rbPos_quantity_seed (controls how many candidates are taken from the sample space, if 1 or smaller than 1, it is used as percentage, else it is used as absolute number)
- vps_toUse (as there can be used multiple algorithms for the SCP of the VPs, it has to be decided with which SCP solution of VPs to work in this phase)
- robtots denavit hartenberg paramters
- outputPath (located at the end of the code, there the output will be saved)

## PHASE 4
### Description
The fourth phase uses the generated robot position candidates and their corresponding reachable VPs as SCP. The least amount of nessesary robot positions need to be found that still cover all VPs. The implemented algorithms to solve the SCP are listed in phase 2.

### Input and configuration
- inputPath (path to a phase 3 output file)
- vps_toUse (as there can be used multiple algorithms for the SCP of the VPs, it has to be decided with which SCP solution of VPs to work in this phase)
- algorithms (need to be commented out, if they should not be applied. Algorithm Paramters are set in file SCP_src)
- outputPath (located at the end of the code, there the output will be saved)
- 
## PHASE 5
### Description
The last phase determines the minimum cost sequences of the given robot positions. Moreover the assigned VPs per robot position are also put in a sequence. For the robot positions the distance between the coordinates is considered as cost. For the VPs the summed joint configuration change to get from one VP to the next is considerd as cost. The phase includes the following steps:
- Step 1: Assign overlapping VPs to the closest robot position. If multiple robot posiiotns cover the same VP, th VPs will only be assinged to the closest robot position. One VP must not be visited multiple times.
- Step 2: Get the robot position sequence via travelling salesman problem (TSP)
- Step 3: Get the VP sequence of each robot positioin via TSP

### Input and configuration
- inputPath (path to a phase 3 output file)
- vps_toUse (as there can be used multiple algorithms for the SCP of the VPs, it has to be decided with which SCP solution of VPs to work in this phase)
- rbPos_toUse (as there can be used multiple algorithms for the SCP of the robot positions, it has to be decided with which SCP solution of robot positions to work in this phase)
- outputPath (located at the end of the code, there the output will be saved)

# Python packages
| Package | Version 
| :-------- | ------|
| absl-py         | 1.1.0 |
| cycler          | 0.11.0 |
| fonttools       | 4.37.1 |
| glfw            | 2.5.3 |
| joblib          | 1.1.0 |
| kiwisolver      | 1.4.4 |
| matplotlib      | 3.5.3 |
| mujoco          | 2.2.0 |
| numpy           | 1.23.1 |
| ortools         | 9.4.1874 |
| packaging       | 21.3 |
| Pillow          | 9.2.0 |
| pip             | 22.1.2 |
| protobuf        | 4.21.5 |
| pyglet          | 1.5.26 |
| PyOpenGL        | 3.1.6 |
| pyparsing       | 3.0.9 |
| python-dateutil | 2.8.2 |
| Rtree           | 1.0.0 |
| scikit-learn    | 1.1.2 |
| scipy           | 1.8.1 |
| setuptools      | 58.1.0 |
| Shapely         | 1.8.2 |
| six             | 1.16.0 |
| threadpoolctl   | 3.1.0 |
| trimesh         | 3.12.7 |
