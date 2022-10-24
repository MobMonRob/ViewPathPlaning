import json
import numpy as np
import scipy
import trimesh
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
import math
import cmath
from math import cos 
from math import sin 
from math import atan2
from math import acos
from math import asin
from math import sqrt
from math import pi
import time
from datetime import datetime



filename = "00_Output Collection/output 221014-080852.json"

with open(filename, 'r') as f:
    dataCollection = json.load(f)
#-----------------
# LEGEND
#-----------------

# "bb"   -> bounding box

gridSize = dataCollection["statistics"]["gridsize (mm)"]
sampleSpace_extension = dataCollection["statistics"]["sampleSpace_extension factor"]
rb_Range = dataCollection["robot"]["rb_operation radius (mm)"]
s_maxRange = dataCollection["sensor"]["s_maxRange (mm)"]
s_size_length = dataCollection["sensor"]["s_size_length (mm)"] 

padWidth_Obj = int(((s_maxRange + s_size_length)*sampleSpace_extension) / gridSize)
tm_rb_maxRange = int(rb_Range / gridSize) + padWidth_Obj
tm_rb_minRange = int(300 / gridSize) # 30 cm mindestabstand zum Objekt

rbPos_quantity_seed = 1000

vps_toUse = "VPs_greedy"


#-----------------
# UR5e
#-----------------
mat = np.matrix
# Denavit Hartenberg Convention Parameters, translated to gridsize 
d = mat([
    (162.5 / gridSize), 
    0.0 , 
    0.0, 
    (133.3 / gridSize),
    (99.7 / gridSize), 
    (99.6 / gridSize)
    ]) 
a = mat([
    0.0, 
    (-425.0 / gridSize), 
    (-392.2 / gridSize), 
    0.0, 
    0.0, 
    0.0]) 

alph = mat([math.pi/2, 0, 0, math.pi/2, -math.pi/2, 0])  

d1 =  d[0,0]
a2 =  a[0,1]
a3 =  a[0,2]
d4 =  d[0,3]
d5 =  d[0,4]
d6 =  d[0,5]


#######################################################################################
#######################################################################################
# HELPER FUNCTIONS

# UR5e FORWARD KINEMATICS - Ryan Keating Johns Hopkins University
# robot base is located at [0,0,0]
def AH(n,th,c  ):

    T_a = mat(np.identity(4), copy=False)
    T_a[0,3] = a[0,n-1]
    T_d = mat(np.identity(4), copy=False)
    T_d[2,3] = d[0,n-1]

    Rzt = mat([[cos(th[n-1,c]), -sin(th[n-1,c]), 0 ,0],
    	         [sin(th[n-1,c]),  cos(th[n-1,c]), 0, 0],
    	         [0,               0,              1, 0],
    	         [0,               0,              0, 1]],copy=False)


    Rxa = mat([[1, 0,                 0,                  0],
    			 [0, cos(alph[0,n-1]), -sin(alph[0,n-1]),   0],
    			 [0, sin(alph[0,n-1]),  cos(alph[0,n-1]),   0],
    			 [0, 0,                 0,                  1]],copy=False)

    A_i = T_d * Rzt * T_a * Rxa
    

    return A_i

def HTrans(th,c ):  
    A_1=AH( 1,th,c  )
    A_2=AH( 2,th,c  )
    A_3=AH( 3,th,c  )
    A_4=AH( 4,th,c  )
    A_5=AH( 5,th,c  )
    A_6=AH( 6,th,c  )

    T_06=A_1*A_2*A_3*A_4*A_5*A_6

    return T_06

# UR5e INVERSE KINEMATICS - Ryan Keating Johns Hopkins University
# robot base is located at [0,0,0]
def invKine(desired_pos):# T60
    th = mat(np.zeros((6, 8)))
    P_05 = (desired_pos * mat([0,0, -d6, 1]).T-mat([0,0,0,1 ]).T)
    
    # **** theta1 ****
    '''theta 1 has a solution in all cases except that d4 > (P05 )xy. 
    this happens when the origin of the 3rd frame is close to the z axis of frame 0. 
    This forms an unreachable cylinder in '''
    '''MODIFIED START'''
    if d4 > sqrt(P_05[2-1,0]*P_05[2-1,0] + P_05[1-1,0]*P_05[1-1,0]):
        return []
    '''MODIFIED END'''

    psi = atan2(P_05[2-1,0], P_05[1-1,0])
    phi = acos(d4 /sqrt(P_05[2-1,0]*P_05[2-1,0] + P_05[1-1,0]*P_05[1-1,0]))
    #The two solutions for theta1 correspond to the shoulder
    #being either left or right
    th[0, 0:4] = pi/2 + psi + phi
    th[0, 4:8] = pi/2 + psi - phi
    th = th.real
    
    # **** theta5 ****
    
    cl = [0, 4]# wrist up or down
    for i in range(0,len(cl)):
        c = cl[i]
        T_10 = np.linalg.inv(AH(1,th,c))
        T_16 = T_10 * desired_pos
        th[4, c:c+2] = + acos((T_16[2,3]-d4)/d6)
        th[4, c+2:c+4] = - acos((T_16[2,3]-d4)/d6)

    th = th.real
    
    # **** theta6 ****
    # theta6 is not well-defined when sin(theta5) = 0 or when T16(1,3), T16(2,3) = 0.

    cl = [0, 2, 4, 6]
    for i in range(0,len(cl)):
        c = cl[i]
        T_10 = np.linalg.inv(AH(1,th,c))
        T_16 = np.linalg.inv( T_10 * desired_pos )
        th[5, c:c+2] = atan2((-T_16[1,2]/sin(th[4, c])),(T_16[0,2]/sin(th[4, c])))
    
    th = th.real

    # **** theta3 ****
    cl = [0, 2, 4, 6]
    for i in range(0,len(cl)):
        c = cl[i]
        T_10 = np.linalg.inv(AH(1,th,c))
        T_65 = AH( 6,th,c)
        T_54 = AH( 5,th,c)
        T_14 = ( T_10 * desired_pos) * np.linalg.inv(T_54 * T_65)
        P_13 = T_14 * mat([0, -d4, 0, 1]).T - mat([0,0,0,1]).T
        t3 = cmath.acos((np.linalg.norm(P_13)**2 - a2**2 - a3**2 )/(2 * a2 * a3)) # norm ?
        th[2, c] = t3.real
        th[2, c+1] = -t3.real

    # **** theta2 and theta 4 ****

    cl = [0, 1, 2, 3, 4, 5, 6, 7]
    for i in range(0,len(cl)):
        c = cl[i]
        T_10 = np.linalg.inv(AH( 1,th,c ))
        T_65 = np.linalg.inv(AH( 6,th,c))
        T_54 = np.linalg.inv(AH( 5,th,c))
        T_14 = (T_10 * desired_pos) * T_65 * T_54
        P_13 = T_14 * mat([0, -d4, 0, 1]).T - mat([0,0,0,1]).T

        # theta 2
        th[1, c] = -atan2(P_13[1], -P_13[0]) + asin(a3* sin(th[2,c])/np.linalg.norm(P_13))
        # theta 4
        T_32 = np.linalg.inv(AH( 3,th,c))
        T_21 = np.linalg.inv(AH( 2,th,c))
        T_34 = T_32 * T_21 * T_14
        th[3, c] = atan2(T_34[1,0], T_34[0,0])

    th = th.real

    return th


#######################################################################################
#######################################################################################


def getRbPosCandidates(_quantity):
    # generate ocupancy matrix; extend the matrix as preparation for the for the dilation step
    obj_bb_ocupancy_matrix = np.ones((dataCollection["obj"]["voxel bounding box"][0], dataCollection["obj"]["voxel bounding box"][1]), dtype=bool)
    obj_bb_ocupancy_matrix_extended = np.pad(obj_bb_ocupancy_matrix, pad_width=(tm_rb_maxRange), mode='constant', constant_values=(0)) # padwidth is the amount of voxel layers to be added -> max range of the sensor 140 mm, voxel size is 10 mm, thus 14 voxel layers have to be added

    # use binary dilation to get the sample space (sampleSpace = maxRange - minRange)
    _dil_ocupancy_matrix_minRange = scipy.ndimage.binary_dilation(obj_bb_ocupancy_matrix_extended, iterations=tm_rb_minRange)
    _dil_ocupancy_matrix_maxRange = scipy.ndimage.binary_dilation(_dil_ocupancy_matrix_minRange, iterations=tm_rb_maxRange-tm_rb_minRange)
    _sampleSpace = np.logical_xor(_dil_ocupancy_matrix_maxRange,_dil_ocupancy_matrix_minRange)

    #_sampleSpace = np.expand_dims(_sampleSpace, axis=2)
    #obj_bb_ocupancy_matrix_extended = np.expand_dims(obj_bb_ocupancy_matrix_extended, axis=2)


    _vp_voxel_coordinates = np.argwhere(_sampleSpace)  - (tm_rb_maxRange - padWidth_Obj) # get ocupied indices in die voxel grid, eqals the viewpoint voxel coordinates
                                                                        # durch das padding wird das obj um die padWidth nach x und y verschoben, um dieursprüngliche position zurückzuerhalten
                                                                        # muss die padWidth von den x und y koordinaten wieder abgezogen werden; Das ursprünglich objekt wurde jedoch in der pos mit pad width belassen, daher muss di wieder drauf gerechtnet werden
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
    dataCollection["statistics"]["init number_of_rbPos_candidates"]    = int(_quantity)
    dataCollection["statistics"]["number_of_rbPos_inSampleSpace"]     = _vp_voxel_count
    for i, vp in enumerate(_vpCandidateCoordinates):
        _vpKey = "rb_pos_" + str(i)
        dataCollection["robot"]["pos"][_vpKey] = {
            "coord" : vp.tolist()
        }

def getRbPosRotation():
    # sensor vieweing direction in initial rb config should look from the obj away (axis aligned)
    # object boundin box starts at the end of the padding of phase 1 pad width
    obj_bb_min = [padWidth_Obj, padWidth_Obj]
    obj_bb_max = [padWidth_Obj + dataCollection["obj"]["voxel bounding box"][0], padWidth_Obj + dataCollection["obj"]["voxel bounding box"][1]]
    for _rbPosKey in list(dataCollection["robot"]["pos"]):
        _rbPos = dataCollection["robot"]["pos"][_rbPosKey]["coord"]
        if _rbPos[1] > obj_bb_max[1]:
            # rotaiton around z axis + 180
            _zRot = 180
            r = R.from_euler('z', 180, degrees=True)
            rMat = r.as_matrix()

        elif _rbPos[1] < obj_bb_min[1]:
            # rotaiton around z axis + 0
            _zRot = 0
            rMat =[
                   [1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]
                   ]

        elif _rbPos[0] < obj_bb_min[0]:
            # rotaiton around z axis - 90
            _zRot = -90
            r = R.from_euler('z', -90, degrees=True)
            rMat = r.as_matrix()

        elif _rbPos[0] > obj_bb_max[0]:
            # rotaiton around z axis + 90
            _zRot = 90
            r = R.from_euler('z', 90, degrees=True)
            rMat = r.as_matrix()
        
        else:
            # invalid position inside the obj
            del dataCollection["robot"]["pos"][_rbPosKey]
            continue # skip the further execution of this iteration

        rb_hom_base_pos =[
               [rMat[0][0], rMat[0][1], rMat[0][2], _rbPos[0]],
               [rMat[1][0], rMat[1][1], rMat[1][2], _rbPos[1]],
               [rMat[2][0], rMat[2][1], rMat[2][2], 0],
               [0, 0, 0, 1]
               ]
        rb_hom_base_pos_inv = np.linalg.inv(rb_hom_base_pos)

        dataCollection["robot"]["pos"][_rbPosKey]["rb_hom_base_pos"] = rb_hom_base_pos
        dataCollection["robot"]["pos"][_rbPosKey]["rb_hom_base_pos_inv"] = rb_hom_base_pos_inv.tolist()
        dataCollection["robot"]["pos"][_rbPosKey]["rb_base_pos_Z_rotation (degree)"] = _zRot


def getPotFieldsPerRbPos():
    
    _vpCoordlst = []
    for _vpKey in list(dataCollection[vps_toUse]):
        _vpCoordlst.append(dataCollection[vps_toUse][_vpKey]["coord"])
    _vpCoordlst = np.array(_vpCoordlst)

    _rbCoordlst = []
    for _rbKey in list(dataCollection["robot"]["pos"]):
        _rbCoordlst.append([dataCollection["robot"]["pos"][_rbKey]["coord"][0], dataCollection["robot"]["pos"][_rbKey]["coord"][1], 0])

    neigh = NearestNeighbors(radius=rb_Range/gridSize, n_jobs=-1, algorithm='kd_tree')
    neigh.fit(_vpCoordlst)

    _rNN = list(neigh.radius_neighbors(_rbCoordlst, return_distance = False))

    for i, _rbKey in enumerate (list(dataCollection["robot"]["pos"])):

        _rNN_idx = [int(_Elem) for _Elem in _rNN[i]]

        if _rNN_idx != []:
            dataCollection["robot"]["pos"][_rbKey]["reachable VPs indices"] = _rNN_idx
        else:
            # all vp voxels are out of range for this rb pos, delete rb pos from the list
            del dataCollection["robot"]["pos"][_rbKey]

def getRobotJointConfiguration():
    _allVPkeys = np.array(list(dataCollection[vps_toUse].keys()))

    for _rbKey in list(dataCollection["robot"]["pos"]):
        dataCollection["robot"]["pos"][_rbKey]["VPs"] = {}
        
        _cur_rb_hom_base_pos_inv = dataCollection["robot"]["pos"][_rbKey]["rb_hom_base_pos_inv"]

        _updatedReachableVPIndices = []

        for _vpIdx in dataCollection["robot"]["pos"][_rbKey]["reachable VPs indices"]:
            _curVpKey = _allVPkeys[_vpIdx]
            _curVProt = dataCollection[vps_toUse][_curVpKey]["RotationMatrix"]
            _curVPccord = dataCollection[vps_toUse][_curVpKey]["coord"]

            _curVPHomTransMat =np.array([
                               [_curVProt[0][0], _curVProt[0][1], _curVProt[0][2], _curVPccord[0]],
                               [_curVProt[1][0], _curVProt[1][1], _curVProt[1][2], _curVPccord[1]],
                               [_curVProt[2][0], _curVProt[2][1], _curVProt[2][2], _curVPccord[2]],
                               [0, 0, 0, 1]
                               ])

            # transform to robot position
            _curVPHomTransMat = np.array(_cur_rb_hom_base_pos_inv).dot(_curVPHomTransMat)

            _curConfigCandidates = invKine(_curVPHomTransMat)

            # If invKine returns false / empty array the Position ist not reachable because it to close to the Z Axis
            if len(_curConfigCandidates) > 0:
                # inv Kine returns 8 Candidate Configs, choose the frist
                _curJntConfig = [_row[0,0] for _row in _curConfigCandidates]

                # check if traget pos is reachable, use first config
                _forwK = HTrans(_curConfigCandidates, 1)
                _eePos = np.array([elem[3] for elem in _forwK.tolist()])
                
                _hom_vp = np.append([_curVPHomTransMat[0,3], _curVPHomTransMat[1,3], _curVPHomTransMat[2,3]],1)

                # as the endeffeoctor pos will not be exctly the target pos it is used the "is_close" function
                if np.allclose(_eePos, _hom_vp, rtol=0.01):
                    dataCollection["robot"]["pos"][_rbKey]["VPs"][_curVpKey] = {"RobotJointConfiguration (radian)" : _curJntConfig} #else: # viewpoint is not reachable, dont add VP to robots list
                    _updatedReachableVPIndices.append(_vpIdx)
        
        if len(_updatedReachableVPIndices) > 0:
            dataCollection["robot"]["pos"][_rbKey]["reachable VPs indices"] = _updatedReachableVPIndices
        else:
            del dataCollection["robot"]["pos"][_rbKey]




#######################################################################################

#######################################################################################

#######################################################################################

#######################################################################################

#######################################################################################

#######################################################################################

#______________________________________________________________________________________
ph = "PHASE 3 STEP 1:"
des = "Get Robot Postion candidates"
#______________________________________________________________________________________
print("\n \n \n============================================== \n", ph, "\n \n", des)

ts=time.time()

getRbPosCandidates(rbPos_quantity_seed)

execTime = time.time()-ts
print("\n", "Time: ", execTime," seconds")

# Save statistics
dataCollection["statistics"]["execTimes"]["PHASE 3 STEP 1: Get Robot Postion candidates - TIME (s)"] = execTime


#______________________________________________________________________________________
ph = "PHASE 3 STEP 2:"
des = "Get Robot Postion orientations"
#______________________________________________________________________________________
print("\n \n \n============================================== \n", ph, "\n \n", des)

ts=time.time()

getRbPosRotation()

execTime = time.time()-ts
print("\n", "Time: ", execTime," seconds")

# Save statistics
dataCollection["statistics"]["execTimes"]["PHASE 3 STEP 2: Get Robot Postion orientations - TIME (s)"] = execTime


#______________________________________________________________________________________
ph = "PHASE 3 STEP 3:"
des = "Get Robot Postion potential fields"
#______________________________________________________________________________________
print("\n \n \n============================================== \n", ph, "\n \n", des)

ts=time.time()

getPotFieldsPerRbPos()

execTime = time.time()-ts
print("\n", "Time: ", execTime," seconds")

# Save statistics
dataCollection["statistics"]["execTimes"]["PHASE 3 STEP 3: Get Robot Postion potential fields - TIME (s)"] = execTime


#______________________________________________________________________________________
ph = "PHASE 3 STEP 4:"
des = "Get robot configurations"
#______________________________________________________________________________________
print("\n \n \n============================================== \n", ph, "\n \n", des)

ts=time.time()

getRobotJointConfiguration()

execTime = time.time()-ts
print("\n", "Time: ", execTime," seconds")

# Save statistics
dataCollection["statistics"]["execTimes"]["PHASE 3 STEP 4: Get robot configurations - TIME (s)"] = execTime


#______________________________________________________________________________________
# Finalizing an Saving results:
#______________________________________________________________________________________
print("\n \n \n============================================== \n",
    "Finalizing...")

ts=time.time()

now = datetime.now()
now_str = now.strftime("%y%m%d-%H%M%S")

filename = "00_Output Collection/PH3 output " + now_str +".json"

with open(filename, 'w') as f:
    json.dump(dataCollection, f, indent=2)


execTime = time.time()-ts
print("\n",
    "Time: ", execTime," seconds \n",
    "Results saved to file <", filename ,"> \n \n")
