import trimesh
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R
from scipy.spatial import Delaunay
import time
from datetime import datetime
import warnings
import math
import json
from sklearn.neighbors import NearestNeighbors


warnings.filterwarnings("ignore", category=UserWarning)


#######################################################################################
#######################################################################################
# GLOBAL CONSTANTS
#-----------------
# LEGEND
#-----------------

# "s"   -> Sensor
# Range -> sensors distance to scanned surface
# "ma"  -> Measurment area , "l" -> lenght, "w" -> width
# "el"  -> edge length from base to top
# "tm"  -> trimesh
# "vp"  -> ViewPoint
# "vxl" -> Voxel
# "obj" -> Object
# "rb"  -> Robot
# "jnt" -> joint

#-----------------
# General 
#-----------------


gridSize = 10 # mm
tm_voxelSize = gridSize / 1000 # -> 10 mm (trimesh voxel size of 1 equals 1 m -> 10 mm / 1000 = 0.01 m)

obj_mesh_scale = 0.0034 # -> "0.0034" results in a blade model length of 1m 
                        # -> "0.0015" results in a can model length of 0.5m 
                        # -> "0.01" results in a cogwheel model length of 0.14m 

number_of_vp_candidates_seed = 0.1 # if 1 or smaller than 1, it is used as percentage, else it is used as absolute number

# dilation procedure führt dazu dass die range nur bei axis alined voxeln dann auch am ende der range entspricht. 
# diagonale voxel haben am ende eine geringere distanz, daher wird der sample space in der max range noch weiter erweitert, damit am ende auch in der diagonalen ausweitung die eigentliche range erreicht wird.
# im potential field step is dadurch jedoch mit VPs zu rechnen die zu weit weg sind (bspw bei den axis aligned voxeln die sind dann ja noch weiter weg) wodurch keine obj surface voxel im potential field sind
sampleSpace_extension = 1.6


# für einen gewünschten overlap je messbereich, kann hier ein verkleinrungsfaktor des Messbereichs angegeben werden. die VPs rücken dadurch automatisch
# durch das Set Covering näher zusammen und der tatsächliche Messbereich weist einen Overlap auf.
ma_scale = 0.985 # -> Messbereich wird um 1.5 % verkleinert. An überlappenden stellen kann die bis zu einer Überlappung 3 % führen


#-----------------
# Sensor Data, "SurfaceControl-3D 3510-80" von Micro-Epsilon
#-----------------
s_size_height = 49 # mm
s_size_width = 220 # mm
s_size_length = 150 # mm

s_minRange = 120 # mm 
s_ma_l_minRange = 67.5 # mm
s_ma_w_minRange = 46 # mm

s_midRange = 130 # mm
s_ma_l_midRange = 80 # mm
s_ma_w_midRange = 50 # mm


s_maxRange = 140 # mm
s_ma_l_maxRange = 77.5 # mm
s_ma_w_maxRange = 52 # mm


# z axis is the optical axis
# y up -> short side of the measurement area (width)
# x sideway -> long side of the measurement area (length)
s_measurementArea = np.array(
                    [
                        [(ma_scale * s_ma_l_minRange/2),       (ma_scale * s_ma_w_minRange/2),        s_minRange],
                        [(ma_scale * s_ma_l_minRange/2),       (ma_scale * s_ma_w_minRange/2)*(-1),   s_minRange],
                        [(ma_scale * s_ma_l_minRange/2)*(-1),  (ma_scale * s_ma_w_minRange/2),        s_minRange],
                        [(ma_scale * s_ma_l_minRange/2)*(-1),  (ma_scale * s_ma_w_minRange/2)*(-1),   s_minRange],

                        [(ma_scale * s_ma_l_midRange/2),       (ma_scale * s_ma_w_midRange/2),        s_midRange],
                        [(ma_scale * s_ma_l_midRange/2),       (ma_scale * s_ma_w_midRange/2)*(-1),   s_midRange],
                        [(ma_scale * s_ma_l_midRange/2)*(-1),  (ma_scale * s_ma_w_midRange/2),        s_midRange],
                        [(ma_scale * s_ma_l_midRange/2)*(-1),  (ma_scale * s_ma_w_midRange/2)*(-1),   s_midRange],

                        [(ma_scale * s_ma_l_maxRange/2),       (ma_scale * s_ma_w_maxRange/2),        s_maxRange],
                        [(ma_scale * s_ma_l_maxRange/2),       (ma_scale * s_ma_w_maxRange/2)*(-1),   s_maxRange],
                        [(ma_scale * s_ma_l_maxRange/2)*(-1),  (ma_scale * s_ma_w_maxRange/2),        s_maxRange],
                        [(ma_scale * s_ma_l_maxRange/2)*(-1),  (ma_scale * s_ma_w_maxRange/2)*(-1),   s_maxRange]
                    ])


#-----------------
# OTHER
#-----------------
# the sensor is mounted on the eeLink -> the robots measurement range = sensor range + sensor size (lenght)
rb_ma_minRange = s_minRange + s_size_length
rb_ma_midRange = s_midRange + s_size_length
rb_ma_maxRange = s_maxRange + s_size_length

rb_measurementArea = np.array(
                    [
                        [s_measurementArea[0, 0],s_measurementArea[0, 1],rb_ma_minRange],
                        [s_measurementArea[1, 0],s_measurementArea[1, 1],rb_ma_minRange],
                        [s_measurementArea[2, 0],s_measurementArea[2, 1],rb_ma_minRange],
                        [s_measurementArea[3, 0],s_measurementArea[3, 1],rb_ma_minRange],

                        [s_measurementArea[4, 0],s_measurementArea[4, 1],rb_ma_midRange],
                        [s_measurementArea[5, 0],s_measurementArea[5, 1],rb_ma_midRange],
                        [s_measurementArea[6, 0],s_measurementArea[6, 1],rb_ma_midRange],
                        [s_measurementArea[7, 0],s_measurementArea[7, 1],rb_ma_midRange],

                        [s_measurementArea[8, 0],s_measurementArea[8, 1],rb_ma_maxRange],
                        [s_measurementArea[9, 0],s_measurementArea[9, 1],rb_ma_maxRange],
                        [s_measurementArea[10, 0],s_measurementArea[10, 1],rb_ma_maxRange],
                        [s_measurementArea[11, 0],s_measurementArea[11, 1],rb_ma_maxRange]
                    ])

# this need to be translated to the gridsize
tm_rb_ma_minRange = rb_ma_minRange / gridSize
tm_rb_ma_midRange = rb_ma_midRange / gridSize
tm_rb_ma_maxRange = rb_ma_maxRange / gridSize

tm_rb_measurementArea = rb_measurementArea / gridSize

# get inclusion distance for potential field. Every voxel of the object within this distance is considered for calculating the mean viewing direction from a specific viewpoint
def getEdgeLengthOfRectangularPyramid(_height, _baseLength, _base_width):
    # s² =  h² + (d/2)²   wobei d = √a² + b²
    _d = (_baseLength**2 + _base_width**2)**0.5
    return (_height**2 + (_d/2)**2)**0.5
tm_rb_ma_inclusionDistance = getEdgeLengthOfRectangularPyramid(rb_ma_maxRange, s_ma_l_maxRange, s_ma_w_maxRange) / gridSize



#######################################################################################
#######################################################################################
#save configuration
dataCollection = {
    "statistics" : {
        "execTimes" : {},
        "gridsize (mm)": gridSize,
        "tm_voxelSize (m)": tm_voxelSize,
        "obj_mesh_scale": obj_mesh_scale,
        "sampleSpace_extension factor": sampleSpace_extension
    },
    "sensor" : {
        "sensor name": "Micro-Epsilon SurfaceControl-3D 3510-80",
        "s_size_height (mm)" : s_size_height,
        "s_size_width (mm)" :  s_size_width,
        "s_size_length (mm)" : s_size_length,

        "s_minRange (mm)" : s_minRange,
        "s_ma_l_minRange (mm)" : s_ma_l_minRange,
        "s_ma_w_minRange (mm)" : s_ma_w_minRange,

        "s_midRange (mm)" :  s_midRange,
        "s_ma_l_midRange (mm)" : s_ma_l_midRange,
        "s_ma_w_midRange (mm)" : s_ma_w_midRange,

        "s_maxRange (mm)" : s_maxRange,
        "s_ma_l_maxRange (mm)" : s_ma_l_maxRange,
        "s_ma_w_maxRange (mm)" : s_ma_w_maxRange,

        "s_measurementArea" : s_measurementArea.tolist()
    },
    "VPs" : {},
    "obj" : {},
    "robot" : {
        "robot name": "UNIVERSAL ROBOTS UR5e",
        "rb_ini_jnt_config": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "rb_operation radius (mm)" : 850,
        "Denavit Hartenberg Parameter" : {
            "d1 (mm)" : 162.5,
            "a2 (mm)" : -425.0,
            "a3 (mm)" : -392.2,
            "d4 (mm)" : 133.3,
            "d5 (mm)" : 99.7,
            "d6 (mm)" : 99.6
        },
        "pos" : {}
    }
}

#######################################################################################
#######################################################################################
# HELPER FUNCTIONS

def line3D(endX,  endY,  endZ,  startX,  startY,  startZ):

    #Output needs to be tople for later comparission within a set, against the potetial field
    # A Fast Voxel Traversal Algorithm for Ray Tracing John Amanatides Andrew Woo
    # with subpixels if there is a diagonal step
    x1 = endX
    y1 = endY
    z1 = endZ
    x0 = startX
    y0 = startY
    z0 = startZ

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)

    
    if x0 < x1:
        stepX = 1
    elif x0 > x1:
        stepX = -1
    else:
        stepX = 0 

    if y0 < y1:
        stepY = 1
    elif y0 > y1:
        stepY = -1
    else:
        stepY = 0   

    if z0 < z1:
        stepZ = 1
    elif z0 > z1:
        stepZ = -1
    else:
        stepZ = 0

    hypotenuse = math.sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2))
    if (dx == 0):
        tMaxX = math.inf
        tDeltaX = math.inf
    else:

        tMaxX = hypotenuse*0.5 / dx
        tDeltaX = hypotenuse / dx

    if (dy == 0):
        tMaxY = math.inf
        tDeltaY = math.inf
    else:
        tMaxY = hypotenuse*0.5 / dy
        tDeltaY = hypotenuse / dy

    if (dz == 0):
        tMaxZ = math.inf
        tDeltaZ = math.inf
    else:
        tMaxZ = hypotenuse*0.5 / dz
        tDeltaZ = hypotenuse / dz



    result = []

    while (x0 != x1 or y0 != y1 or z0 != z1):
        if (tMaxX < tMaxY):
            if (tMaxX < tMaxZ):
                x0 = x0 + stepX
                tMaxX = tMaxX + tDeltaX
            
            elif (tMaxX > tMaxZ):
                z0 = z0 + stepZ
                tMaxZ = tMaxZ + tDeltaZ
            
            else:
                # this is the diagonal case
                x0 = x0 + stepX
                tMaxX = tMaxX + tDeltaX
                z0 = z0 + stepZ
                tMaxZ = tMaxZ + tDeltaZ

                #also add the sub pixels
                result.extend([(x0-stepX,y0,z0), (x0,y0,z0-stepZ)])
            
        
        elif (tMaxX > tMaxY):
            if (tMaxY < tMaxZ):
                y0 = y0 + stepY
                tMaxY = tMaxY + tDeltaY
            
            elif (tMaxY > tMaxZ):
                z0 = z0 + stepZ
                tMaxZ = tMaxZ + tDeltaZ
            
            else:
                # this is the diagonal case
                y0 = y0 + stepY
                tMaxY = tMaxY + tDeltaY
                z0 = z0 + stepZ
                tMaxZ = tMaxZ + tDeltaZ

                #also add the sub pixels
                result.extend([(x0,y0-stepY,z0), (x0,y0,z0-stepZ)])
          
        
        else:
            if (tMaxY < tMaxZ):
                y0 = y0 + stepY
                tMaxY = tMaxY + tDeltaY
                x0 = x0 + stepX
                tMaxX = tMaxX + tDeltaX
            
            elif (tMaxY > tMaxZ):
                z0 = z0 + stepZ
                tMaxZ = tMaxZ + tDeltaZ
            
            else:
                # this is the diagonal case
                x0 = x0 + stepX
                tMaxX = tMaxX + tDeltaX
                y0 = y0 + stepY
                tMaxY = tMaxY + tDeltaY
                z0 = z0 + stepZ
                tMaxZ = tMaxZ + tDeltaZ

                #also add the sub pixels
                result.extend([(x0-stepX,y0,z0), (x0,y0-stepY,z0), (x0,y0,z0-stepZ)])
        
        result.append((x0,y0,z0))

    return(result)

def get_rotation_matrix(_vec2, _vec1=np.array([0.0, 0.0, 1.0])):
    # get matrix with scipy
    # default for the z axis (which is the optical axis)
    _vec1 = np.reshape(_vec1, (1, -1))
    _vec2 = np.reshape(_vec2, (1, -1))

    _r = R.align_vectors(_vec2, _vec1)

    return _r[0].as_matrix()

def exportVoxelmodel(_time):
    _sceneGrid = np.array(dataCollection["obj"]["voxel_ocupancy_matrix_extended"])
    now_str = _time

    filename = "00_Output Collection/bladeVoxelized "+str(gridSize)+"mm " + now_str +".stl"

    _gridAsMesh = trimesh.voxel.base.VoxelGrid(_sceneGrid).as_boxes()
    _gridAsMesh.export(filename) 


#######################################################################################
#######################################################################################
# MAIN FUNCTIONS

def voxelizeObject(_objMesh):
    # voxelize model via trimesh
    _objVoxelized = _objMesh.voxelized(tm_voxelSize, method='subdivide', max_iter=1000)
    
    # reduce VoxelGrid voxelcount by removing non surface voxels
    _objVoxelized.fill(method='orthographic')
    _objVoxelized.hollow()

    #_objVoxelized.show(flags={'wireframe': True, 'axis': True})

    # generate ocupancy matrix; extend the matrix as preparation for the for the dilation step
    _obj_voxel_ocupancy_matrix = np.asanyarray(_objVoxelized.matrix, dtype=bool)
    _obj_voxel_ocupancy_matrix_extended = np.pad(_obj_voxel_ocupancy_matrix, pad_width=(int(tm_rb_ma_maxRange*sampleSpace_extension)), mode='constant', constant_values=(0)) # padwidth is the amount of voxel layers to be added -> max range of the sensor 140 mm, voxel size is 10 mm, thus 14 voxel layers have to be added
    _obj_voxel_coordinates = np.argwhere(_obj_voxel_ocupancy_matrix_extended) # get ocupied indices in die voxel grid, eqals the object voxel coordinates
    
    # save results
    dataCollection["statistics"]["Occupied Object Voxel Count"] = int(_obj_voxel_coordinates.shape[0])
    dataCollection["obj"]["voxel_ocupancy_matrix_extended"] = _obj_voxel_ocupancy_matrix_extended.tolist()
    dataCollection["obj"]["voxel coordinates"] = _obj_voxel_coordinates.tolist()
    dataCollection["obj"]["voxel bounding box"] = list(_obj_voxel_ocupancy_matrix.shape)
    
def getViewPointCandidates(_quantity):
    _objOcupancyMatrix = dataCollection["obj"]["voxel_ocupancy_matrix_extended"]
    # use binary dilation to get the sample space (sampleSpace = maxRange - minRange)
    _dil_ocupancy_matrix_minRange = scipy.ndimage.binary_dilation(_objOcupancyMatrix, iterations=int(tm_rb_ma_minRange))
    _dil_ocupancy_matrix_maxRange = scipy.ndimage.binary_dilation(_dil_ocupancy_matrix_minRange, iterations=int(tm_rb_ma_maxRange*sampleSpace_extension - tm_rb_ma_minRange))
    _sampleSpace = np.logical_xor(_dil_ocupancy_matrix_maxRange,_dil_ocupancy_matrix_minRange)

    _vp_voxel_coordinates = np.argwhere(_sampleSpace) # get ocupied indices in die voxel grid, eqals the viewpoint voxel coordinates
    _vp_voxel_count = int(_vp_voxel_coordinates.shape[0]) 
    _vp_voxel_listIndices = np.arange(0,_vp_voxel_count) # equals list of indices of the coordinate list

    if _quantity <= 1: # if smaller than 1, it is used as percentage
        _quantity = _vp_voxel_count*_quantity 

    _vp_voxel_candidate_listIndices = np.random.choice(_vp_voxel_listIndices, int(_quantity), replace=False) # choose random indices as candidates
    _vpCandidateCoordinates = _vp_voxel_coordinates[_vp_voxel_candidate_listIndices]
    
    #TEsTING###########
    #_vpCandidateCoordinates = _vp_voxel_coordinates[200000:300000]
    #TEsTING###########

    # save result
    dataCollection["statistics"]["init number_of_vp_candidates"]    = int(_quantity)
    dataCollection["statistics"]["number_of_vps_inSampleSpace"]     = _vp_voxel_count
    for i, vp in enumerate(_vpCandidateCoordinates):
        _vpKey = "vp_" + str(i)
        dataCollection["VPs"][_vpKey] = {
            "coord" : vp.tolist()
        }

def getRotationMatrixPerVP():
    _objVoxelCoordinates = np.array(dataCollection["obj"]["voxel coordinates"])

    neigh = NearestNeighbors(radius=tm_rb_ma_inclusionDistance, n_jobs=-1, algorithm='kd_tree')
    neigh.fit(_objVoxelCoordinates)

    # using a list that is processed at once is faster than using a loop that is calling the method each element
    # test with VP = 34.000 and ObjVx = 7500
    #   - loop -> 20 sec
    #   - list and one function call -> 0.28 sec
    _vpCoordlst = []
    for _vpKey in list(dataCollection["VPs"]):
        _vpCoordlst.append(dataCollection["VPs"][_vpKey]["coord"])

    _rNN = list(neigh.radius_neighbors(_vpCoordlst, return_distance = False))


    for i, _vpKey in enumerate (list(dataCollection["VPs"])):

        _rNN_idx = [int(_Elem) for _Elem in _rNN[i]]

        if _rNN_idx != []:
            _objVxlCoords = _objVoxelCoordinates[_rNN_idx].tolist()

            dataCollection["VPs"][_vpKey]["potential Field Obj Voxel coordinates"]  = _objVxlCoords
            dataCollection["VPs"][_vpKey]["potential Field Obj Voxel Indices"]      = _rNN_idx
            dataCollection["VPs"][_vpKey]["visible Obj Voxel coordinates"]          = _objVxlCoords
            dataCollection["VPs"][_vpKey]["visible Obj Voxel Indices"]              = _rNN_idx

            _TEMP_VPObj_Vec = []
            for _objVxl in _objVxlCoords:
                _TEMP_VPObj_Vec.append(np.array(_objVxl) - np.array(dataCollection["VPs"][_vpKey]["coord"]))
            
            _TEMP_Vec = np.sum(_TEMP_VPObj_Vec,  axis=0)
            _TEMP_Norm = np.linalg.norm(_TEMP_Vec)        
            _meanViewVirectionFromCurVP = (_TEMP_Vec/_TEMP_Norm) # als Einheitsvektor
            _rotationMatrixCurVP = get_rotation_matrix(_meanViewVirectionFromCurVP)
            _rotationMatrixCurVPinv = np.linalg.inv(_rotationMatrixCurVP)

            # TESTING Start ######################
            # mat = _rotationMatrixCurVP
            # vec1 = np.array([0.0, 0.0, 1.0])
            # vec2 = _meanViewVirectionFromCurVP
            # vec1_rot = mat.dot(vec1)
            # assert np.allclose(vec1_rot/np.linalg.norm(vec1_rot), vec2/np.linalg.norm(vec2))
            # TESTING End ######################

            dataCollection["VPs"][_vpKey]["RotationMatrix"]           = _rotationMatrixCurVP.tolist()
            dataCollection["VPs"][_vpKey]["RotationMatrixInv"]        = _rotationMatrixCurVPinv.tolist()

        else:
            # all obj voxels are out of range for this viewpoint, delete viewpoint from the list
            del dataCollection["VPs"][_vpKey]

def getObjectVoxelsInMeasurmentAreaPerVP():

    for _vpKey in list(dataCollection["VPs"]):
        # use the mesurement area (polyhedra (german:Polyeder), which is furtunatly convex) for the visibility matrix
        # Calculate mesurement area from VP and its viewing direction
        _maCurVP = tm_rb_measurementArea.dot(dataCollection["VPs"][_vpKey]["RotationMatrixInv"]) + dataCollection["VPs"][_vpKey]["coord"]
        _objVxlinMA_binary = Delaunay(_maCurVP).find_simplex(dataCollection["VPs"][_vpKey]["potential Field Obj Voxel coordinates"]) >=0
        _objVxlinMA_indices = np.argwhere(_objVxlinMA_binary).reshape(-1) # das sind aus der Potential field Liste die Indices, welche in der mesurement area sind
        
        if _objVxlinMA_indices.tolist() != []:
            _updatedObjVxlCoord = np.array(dataCollection["VPs"][_vpKey]["potential Field Obj Voxel coordinates"])[_objVxlinMA_indices].tolist()
            _updatedObjVxlIdx   = np.array(dataCollection["VPs"][_vpKey]["potential Field Obj Voxel Indices"])[_objVxlinMA_indices].tolist()
            dataCollection["VPs"][_vpKey]["Measurement Area Obj Voxel coordinates"]     = _updatedObjVxlCoord
            dataCollection["VPs"][_vpKey]["Measurement Area Obj Voxel Indices"]         = _updatedObjVxlIdx
            dataCollection["VPs"][_vpKey]["visible Obj Voxel coordinates"]              = _updatedObjVxlCoord
            dataCollection["VPs"][_vpKey]["visible Obj Voxel Indices"]                  = _updatedObjVxlIdx
        else:
            # all obj voxels are outside the Measurement area of this viewpoint, delete viewpoint from the list
            del dataCollection["VPs"][_vpKey]

def cullOccludedVoxels():
  
    for _vpKey in list(dataCollection["VPs"]):
        _curVPPotField_asSet = set([tuple(_coordElem) for _coordElem in dataCollection["VPs"][_vpKey]["potential Field Obj Voxel coordinates"]])

        _curVPVisObjVx = []
        _curVPVisObjVx_idx = []

        for j, _objVx in enumerate(dataCollection["VPs"][_vpKey]["Measurement Area Obj Voxel coordinates"]):
            vxlsOnRay = line3D(_objVx[0], _objVx[1], _objVx[2], dataCollection["VPs"][_vpKey]["coord"][0], dataCollection["VPs"][_vpKey]["coord"][1], dataCollection["VPs"][_vpKey]["coord"][2])

            _objVx = tuple(_objVx)

            # check if any of voxels on the ray are in the potential field, if there is one the loop can already stop
            _curVxIsOccluded = False
            for vx in vxlsOnRay:
                # all datatypes need to be tuple here, the pot field list needs to be a set for better performance           
                if vx in _curVPPotField_asSet and vx != _objVx:
                    _curVxIsOccluded=True
                    break
                
            if _curVxIsOccluded == False:
                _curVPVisObjVx.append(_objVx)
                _curVPVisObjVx_idx.append(dataCollection["VPs"][_vpKey]["Measurement Area Obj Voxel Indices"][j])
            
       
        if _curVPVisObjVx_idx != []:
            dataCollection["VPs"][_vpKey]["visible Obj Voxel coordinates"]  = _curVPVisObjVx
            dataCollection["VPs"][_vpKey]["visible Obj Voxel Indices"]      = _curVPVisObjVx_idx
        else:
            # all obj voxels of this viewpoint are occluded, delete viewpoint from the list
            del dataCollection["VPs"][_vpKey]


#######################################################################################

#######################################################################################

#######################################################################################

#######################################################################################

#######################################################################################

#######################################################################################


#______________________________________________________________________________________
# INITIALISATION:
# Load object model. Input type can be Point Cloud, Mesh, etc, from initial scan of the operation area, with eg two cameras
#______________________________________________________________________________________
print("\n \n \n============================================== \n",
    "Initializing...")

ts=time.time()

obj_mesh_path = 'C:/Users/nehem/OneDrive/Studium/Master/_wissensch. Arbeiten/04_Masterarbeit/06_mujocoModels/meshes/01_blade.stl'
#obj_mesh_path = 'C:/Users/nehem/OneDrive/Studium/Master/_wissensch. Arbeiten/04_Masterarbeit/06_mujocoModels/meshes/02_watering_can_thickshell.stl'
#obj_mesh_path = 'C:/Users/nehem/OneDrive/Studium/Master/_wissensch. Arbeiten/04_Masterarbeit/06_mujocoModels/meshes/03_cogwheel.stl'


obj_mesh = trimesh.load(obj_mesh_path).apply_scale(obj_mesh_scale)

#rotation des Objektes
r = R.from_euler('y', 90, degrees=True)
rMat = r.as_matrix()
rMathom =[
            [rMat[0][0], rMat[0][1], rMat[0][2], 0],
            [rMat[1][0], rMat[1][1], rMat[1][2], 0],
            [rMat[2][0], rMat[2][1], rMat[2][2], 0],
            [0, 0, 0, 1]
        ]
rMathom_inv = np.linalg.inv(rMathom)
obj_mesh.apply_transform(rMathom_inv)
#obj_mesh.show(flags={'wireframe': True, 'axis': True})

execTime = time.time()-ts
print("\n","Object mesh loaded in: ", execTime," seconds")

# Save statistics
dataCollection["statistics"]["execTimes"]["INITIALISATION: Object mesh loaded - TIME (s)"] = execTime

obj_surfaceArea = obj_mesh.area * 1000000
dataCollection["obj"]["obj_mesh surface area (mm²)"] = obj_surfaceArea
dataCollection["obj"]["s_minRange theor. scans to fully cover obj"] = obj_surfaceArea / (s_ma_l_minRange * s_ma_w_minRange)
dataCollection["obj"]["s_midRange theor. scans to fully cover obj"] = obj_surfaceArea / (s_ma_l_midRange * s_ma_w_midRange)
dataCollection["obj"]["s_maxRange theor. scans to fully cover obj"] = obj_surfaceArea / (s_ma_l_maxRange * s_ma_w_maxRange)

obj_boundingbox = obj_mesh.bounds
dataCollection["obj"]["size xyz (cm)"] = [
    (abs(obj_boundingbox[0,0]) + abs(obj_boundingbox[1,0])) * 100,
    (abs(obj_boundingbox[0,1]) + abs(obj_boundingbox[1,1])) * 100,
    (abs(obj_boundingbox[0,2]) + abs(obj_boundingbox[1,2])) * 100
]



#______________________________________________________________________________________
ph = "PHASE 1 STEP 1:"
des = "Generate Extended Object Voxel Grid"
# Voxelize object model as ocupancy matrix. Extend the matrix as preparation for the dilation step.
#______________________________________________________________________________________
print("\n \n \n============================================== \n", ph, "\n \n", des)

ts=time.time()

voxelizeObject(obj_mesh)

execTime = time.time()-ts
print("\n", "Time: ", execTime," seconds")

# Save statistics
dataCollection["statistics"]["execTimes"]["PHASE 1 STEP 1: Voxelize Obj - TIME (s)"] = execTime



#_gridAsMesh2 = trimesh.voxel.base.VoxelGrid(np.array(dataCollection["obj"]["voxel_ocupancy_matrix_extended"])).as_boxes()
#_gridAsMesh2.visual.face_colors = [255, 255, 0, 255]
#_gridAsMesh2.show(flags={'axis': True})



#______________________________________________________________________________________
ph = "PHASE 1 STEP 2:"
des = "Get ViewPoint Candidates"
# Generate viepoint sample space via binary dilation. Get viewpoint candidates via Gaussian distribution
#______________________________________________________________________________________
print("\n \n \n============================================== \n", ph, "\n \n", des)

ts=time.time()

getViewPointCandidates(number_of_vp_candidates_seed)

execTime = time.time()-ts
print("\n", "Time: ", execTime," seconds")

# Save statistics
dataCollection["statistics"]["execTimes"]["PHASE 1 STEP 2: Get ViewPoint Candidates - TIME (s)"] = execTime



#______________________________________________________________________________________
ph = "PHASE 1 STEP 3:"
des = "Potential Fields and Rotation Matrices per VP"
# Calculate object voxels that are within the "potential field" of the viewpoints. 
# Use the voxels of the potential field to calculate mean viewing direction of the viepoints
#______________________________________________________________________________________
print("\n \n \n============================================== \n", ph, "\n \n", des)

ts=time.time()

getRotationMatrixPerVP()

execTime = time.time()-ts
print("\n", "Time: ", execTime," seconds")

# Save statistics
dataCollection["statistics"]["execTimes"]["PHASE 1 STEP 3: Pot. Field & Rot. Matrices per VP - TIME (s)"] = execTime


#______________________________________________________________________________________
ph = "PHASE 1 STEP 4:"
des = "Calculate object voxels that are within the measurement area of the viewpoints"
# Only obj voxels in the potential field are used per VP in terms of dimensionality reduction
#______________________________________________________________________________________
print("\n \n \n============================================== \n", ph, "\n \n", des)

ts=time.time()

getObjectVoxelsInMeasurmentAreaPerVP()

execTime = time.time()-ts
print("\n", "Time: ", execTime," seconds")

# Save statistics
dataCollection["statistics"]["execTimes"]["PHASE 1 STEP 4: Obj Vxl in MA per VP - TIME (s)"] = execTime


#______________________________________________________________________________________
ph = "PHASE 1 STEP 5:"
des = "Ray tracing for ocllusion culling per Object Voxels in MA per VP"
#______________________________________________________________________________________
print("\n \n \n============================================== \n", ph, "\n \n", des)

ts=time.time()

cullOccludedVoxels()

execTime = time.time()-ts
print("\n", "Time: ", execTime," seconds")

# Save statistics
dataCollection["statistics"]["execTimes"]["PHASE 1 STEP 5: ocllusion culling per VP - TIME (s)"] = execTime



#______________________________________________________________________________________
# Finalizing an Saving results:
#______________________________________________________________________________________
print("\n \n \n============================================== \n",
    "Finalizing...")

ts=time.time()

# clean up for output (only visible Voxels are relevant)
for _vpKey in list(dataCollection["VPs"]):
    del dataCollection["VPs"][_vpKey]["potential Field Obj Voxel coordinates"]
    del dataCollection["VPs"][_vpKey]["potential Field Obj Voxel Indices"]
    del dataCollection["VPs"][_vpKey]["Measurement Area Obj Voxel coordinates"]
    del dataCollection["VPs"][_vpKey]["Measurement Area Obj Voxel Indices"]
    del dataCollection["VPs"][_vpKey]["visible Obj Voxel coordinates"]

now = datetime.now()
now_str = now.strftime("%y%m%d-%H%M%S")

filename = "00_Output Collection/PH1 output " + now_str +".json"

with open(filename, 'w') as f:
    json.dump(dataCollection, f, indent=2)

#exportVoxelmodel(now_str)

execTime = time.time()-ts
print("\n",
    "Time: ", execTime," seconds \n",
    "Results saved to file <", filename ,"> \n \n")

