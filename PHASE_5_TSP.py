import json
import numpy as np
import trimesh
import random
import scipy
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import time
from datetime import datetime




filename = "00_Output Collection/output 221014-091602.json"


with open(filename, 'r') as f:
    dataCollection = json.load(f)

vps_toUse = "VPs_greedy"
rbPos_toUse = "rbPos_greedy"

def assignOverlapVPs():
 
    rb_Pos = dataCollection["robot"][rbPos_toUse]

    occurLst = {}
    _allVPkeys = list(dataCollection[vps_toUse].keys())

    for _curRbPos_key in list(rb_Pos):
        for _VPsIdx in rb_Pos[_curRbPos_key]["reachable VPs indices"]:
            if _VPsIdx not in occurLst:
                occurLst[_VPsIdx] = []

            _distVec = np.array([rb_Pos[_curRbPos_key]["coord"][0], rb_Pos[_curRbPos_key]["coord"][1],0]) - np.array(dataCollection[vps_toUse][_allVPkeys[_VPsIdx]]["coord"])
            _dist = np.linalg.norm(_distVec)
            occurLst[_VPsIdx].append([_curRbPos_key, _dist])
    
    def getElemCost(elem):
        return elem[1]
    
    for _VPsIdx in occurLst:
        if len(occurLst[_VPsIdx]) > 1:
            _minDistRb = min(occurLst[_VPsIdx], key=getElemCost)

            for _curRbPos_key in occurLst[_VPsIdx]:
                if _curRbPos_key[0] != _minDistRb[0]:
                    rb_Pos[_curRbPos_key[0]]["reachable VPs indices"].remove(_VPsIdx)
                    del rb_Pos[_curRbPos_key[0]]["VPs"][_allVPkeys[_VPsIdx]]

def getTravelCostMatrix(_jntConfigs, _mode):
    _dimensions = len(_jntConfigs)
    _travelCostMatrix = np.zeros((_dimensions,_dimensions), dtype=np.int32)

    match _mode:
        case 0:
           
            for i, _jntConf_1 in enumerate(_jntConfigs):
                for j in range(i, len(_jntConfigs)):
                    _jntConf_2 = _jntConfigs[j]

                    if _jntConf_1 != _jntConf_2:
                        _TEMP_curCost = []
                        for k in range(0,len(_jntConf_1)):
                            _TEMP_curCost.append(abs(_jntConf_1[k]-_jntConf_2[k]))
                        
                        _sumCost = int(sum(_TEMP_curCost)*1000000) # googls TSP Algorithems takes only integers as input, -> *1.000.000 to still get the decimal placesin the number

                        _travelCostMatrix[i,j] = _sumCost
                        _travelCostMatrix[j,i] = _sumCost

            
        case 1:
            for i, _jntConf_1 in enumerate(_jntConfigs):
                for j in range(i, len(_jntConfigs)):
                    _jntConf_2 = _jntConfigs[j]

                    if _jntConf_1 != _jntConf_2:
                        _TEMP_curCost = []
                        for k in range(0,len(_jntConf_1)):
                            _TEMP_curCost.append(abs(_jntConf_1[k]-_jntConf_2[k]))
                        # die kosten sollen die dauer von der einen zur anderen Pos sein. 
                        # zunächst wurde hier die Summe der gelenkwinkel änderungen genommen
                        # da die gelenkwinkeländerungen jedoch nicht nacheinader sondern gleichzeitig passieren, steigt nicht die dauer der bewegungn wenn meherere Gelenke geänert werden müssen
                        # Die dauer der bewegung dauert maximal so lange wie für die gröte Winkeländerung notwendigist
                        # daher wird nun die max Winkeländerung verwendet
                        _sumCost = int(max(_TEMP_curCost)*1000000) # googls TSP Algorithems takes only integers as input, -> *1.000.000 to still get the decimal placesin the number

                        _travelCostMatrix[i,j] = _sumCost
                        _travelCostMatrix[j,i] = _sumCost   

    return _travelCostMatrix.tolist()

def getTravelCostMatrix_byVpDist(_VPcoords):
    _dimensions = len(_VPcoords)
    _travelCostMatrix = np.zeros((_dimensions,_dimensions), dtype=np.int32)

    for i, _jntConf_1 in enumerate(_VPcoords):
        for j in range(i, len(_VPcoords)):
            _jntConf_2 = _VPcoords[j]

            if np.array_equal(_jntConf_1,_jntConf_2) == False:
                _distVec = np.array([_jntConf_2[0], _jntConf_2[1]]) - np.array([_jntConf_1[0], _jntConf_1[1]])
                _Cost = np.linalg.norm(_distVec)
                _travelCostMatrix[i,j] = _Cost
                _travelCostMatrix[j,i] = _Cost

    return _travelCostMatrix.tolist()

def get_routes(solution, routing, manager):
    #FROM: https://developers.google.com/optimization/routing/tsp
    """Get vehicle routes from a solution and store them in an array."""
    # Get vehicle routes and store them in a two dimensional array whose
    # i,j entry is the jth location visited by vehicle i along its route.
    routes = []
    for route_nbr in range(routing.vehicles()):
        index = routing.Start(route_nbr)
        route = [manager.IndexToNode(index)]
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))
        routes.append(route)
    return routes

def solveTSP(_costMatrix):
        # TSP ###################################
        #########################################
        #FROM: https://developers.google.com/optimization/routing/tsp
        # Instantiate the data problem. Store the data for the problem
        tsp_data = {}
        tsp_data['distance_matrix'] = _costMatrix
        tsp_data['num_vehicles'] = 1
        tsp_data['depot'] = 0           # element at pos 0 is the start and end (in our case it is the robots ini jnt config)

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(tsp_data['distance_matrix']), tsp_data['num_vehicles'], tsp_data['depot'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)


        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return tsp_data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        # get the sequence
        seq_idx = get_routes(solution, routing, manager)[0] # only one vehicle is used in our case, -> always select the first entry 

        return seq_idx

def getRobotJointConfigurationSequence():
    for _rbPosKey in list(dataCollection["robot"][rbPos_toUse]):
            
        # prepare cost matrix ###################
        #########################################
        # add initial robot configuration to the sequence
        rb_jnt_configs = []
        
        rb_jnt_configs.append(dataCollection["robot"]["rb_ini_jnt_config"])

        _VPs = list(dataCollection["robot"][rbPos_toUse][_rbPosKey]["VPs"])

        for _vpKey in _VPs:
            rb_jnt_configs.extend([dataCollection["robot"][rbPos_toUse][_rbPosKey]["VPs"][_vpKey]["RobotJointConfiguration (radian)"]])

        _travelCostMatrix = getTravelCostMatrix(rb_jnt_configs, 0)


        ## ALTERNATIVE ##########################
        #rb_vp_coords = []
        #rb_vp_coords.append(np.array([dataCollection["robot"][rbPos_toUse][_rbPosKey]["rb_hom_vp_home_pos"][0][3], dataCollection["robot"][rbPos_toUse][_rbPosKey]["rb_hom_vp_home_pos"][1][3],dataCollection["robot"][rbPos_toUse][_rbPosKey]["rb_hom_vp_home_pos"][2][3]]))
        #for _vpKey in _VPs:
        #    rb_vp_coords.extend(np.array([dataCollection["VPs"][_vpKey]["coord"]]))
        #
        #_travelCostMatrix = getTravelCostMatrix_byVpDist(rb_vp_coords)
        ## ALTERNATIVE ##########################


        seq_idx = solveTSP(_travelCostMatrix)


        # add config sequece info to output (the sequence contains start an end elent, which is the robots ini jnt config, that is not in 
        # the robots vp set -> thus remove first and last entry and use "indices - 1" to match the vp indices of the robots)
        
        seq_idx = seq_idx[1:-1] # remove first and last elem

        for i, key in enumerate(seq_idx):
            dataCollection["robot"][rbPos_toUse][_rbPosKey]["VPs"][_VPs[key-1]]["sequence number"] = i
            
def getRobotPositionSequence():
    rb_coords = []
    all_rbPosKeys = []
    for _rbPosKey in list(dataCollection["robot"][rbPos_toUse]):
            
        # prepare cost matrix ###################
        #########################################
        # add initial robot configuration to the sequence
        all_rbPosKeys.append(_rbPosKey)
        rb_coords.append(tuple(dataCollection["robot"][rbPos_toUse][_rbPosKey]["coord"]))


    _travelCostMatrix = getTravelCostMatrix_byVpDist(rb_coords)

    seq_idx = solveTSP(_travelCostMatrix)
    
    seq_idx = seq_idx[0:-1] # remove last element, as it is the starting point, the first element will not be removed as no base position is used hat is added in front(as e.g. in the config sequece)

    for i, key in enumerate(seq_idx):
        dataCollection["robot"][rbPos_toUse][all_rbPosKeys[key]]["sequence number"] = i





#______________________________________________________________________________________
ph = "PHASE 5 STEP 1:"
des = "assignOverlapVPs"
#______________________________________________________________________________________
print("\n \n \n============================================== \n", ph, "\n \n", des)

ts=time.time()

assignOverlapVPs()

execTime = time.time()-ts
print("\n", "Time: ", execTime," seconds")

# Save statistics
dataCollection["statistics"]["execTimes"]["PHASE 5 STEP 1: assignOverlapVPs - TIME (s)"] = execTime



#______________________________________________________________________________________
ph = "PHASE 5 STEP 2:"
des = "getRobotJointConfigurationSequence"
#______________________________________________________________________________________
print("\n \n \n============================================== \n", ph, "\n \n", des)

ts=time.time()

getRobotJointConfigurationSequence()

execTime = time.time()-ts
print("\n", "Time: ", execTime," seconds")

# Save statistics
dataCollection["statistics"]["execTimes"]["PHASE 5 STEP 2: getRobotJointConfigurationSequence - TIME (s)"] = execTime





#______________________________________________________________________________________
ph = "PHASE 5 STEP 3:"
des = "getRobotPositionSequence"
#______________________________________________________________________________________
print("\n \n \n============================================== \n", ph, "\n \n", des)

ts=time.time()

getRobotPositionSequence()

execTime = time.time()-ts
print("\n", "Time: ", execTime," seconds")

# Save statistics
dataCollection["statistics"]["execTimes"]["PHASE 5 STEP 3: getRobotPositionSequence - TIME (s)"] = execTime


#______________________________________________________________________________________
# Finalizing an Saving results:
#______________________________________________________________________________________
print("\n \n \n============================================== \n",
    "Finalizing...")

ts=time.time()


now = datetime.now()
now_str = now.strftime("%y%m%d-%H%M%S")

filename = "00_Output Collection/PH5 output " + now_str +".json"

with open(filename, 'w') as f:
    json.dump(dataCollection, f, indent=2)


execTime = time.time()-ts
print("\n",
    "Time: ", execTime," seconds \n",
    "Results saved to file <", filename ,"> \n \n")

