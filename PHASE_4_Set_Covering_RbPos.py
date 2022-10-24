import time
from datetime import datetime
import warnings
import json
import copy
import random
import math
import numpy as np
import itertools
import SCP_src as scp


warnings.filterwarnings("ignore", category=UserWarning)


filename = "00_Output Collection/output 221014-082103.json"

with open(filename, 'r') as f:
    dataCollection = json.load(f)


vps_toUse = "VPs_greedy"

#______________________________________________________________________________________
ph = "PHASE 4 STEP 1:"
des = "Set Covering get minimum required RbPos"
#______________________________________________________________________________________
print("\n \n \n============================================== \n", ph, "\n \n", des)

ts=time.time()


subsets = []
for _vpKey in list(dataCollection["robot"]["pos"]):
    subsets.append(set(dataCollection["robot"]["pos"][_vpKey]["reachable VPs indices"]))

universe = set()
for s in subsets:
    universe = universe.union(s)

_coveredObjVxls_unique = len(universe)

dataCollection["statistics"]["RbPos_setCovering"]={
    "Global Coverage (%)" : (_coveredObjVxls_unique/len(dataCollection[vps_toUse]))*100
}
    
_allVPkeys_before = list(dataCollection["robot"]["pos"].keys())

scpResults = {}


ts=time.time()
result_idx_greedy = scp.greedySetCovering(universe, subsets)
execTime_greedy = time.time()-ts
greedySolutionEncoding = scp.encodeSolution(result_idx_greedy, subsets)
scpResults["greedy"] = {
        "idx":result_idx_greedy,
        "enc":scp.encodeSolution(result_idx_greedy, subsets),
        "execTime":execTime_greedy
    }

ts=time.time()
result_idx_naive = scp.naiveSetCovering(universe, subsets)
execTime_naive = time.time()-ts
scpResults["naive"]= {
        "idx":result_idx_naive,
        "enc":scp.encodeSolution(result_idx_naive, subsets),
        "execTime":execTime_naive
        }
#
#ts=time.time()
#SASolutionEncoding = scp.simulatedAnnealingSetCovering(universe, subsets)
#execTime_SA = time.time()-ts
#scpResults["SA"]= {
#        "idx":scp.decodeSolution(SASolutionEncoding),
#        "enc":SASolutionEncoding,
#        "execTime":execTime_SA
#    }
#
#ts=time.time()
#PSOSolutionEncoding = scp.particleSwarmSetCovering(universe, subsets)
#execTime_PSO = time.time()-ts
#scpResults["PSO"]= {
#        "idx":scp.decodeSolution(PSOSolutionEncoding),
#        "enc":PSOSolutionEncoding,
#        "execTime":execTime_PSO
#    }
#
#ts=time.time()
#PSO_GreedySolutionEncoding = scp.particleSwarmSetCovering(universe, subsets, greedySolutionEncoding)
#execTime_PSO_Greedy = time.time()-ts
#scpResults["PSO_Greedy"]= {
#        "idx":scp.decodeSolution(PSO_GreedySolutionEncoding),
#        "enc":PSO_GreedySolutionEncoding,
#        "execTime":execTime_PSO_Greedy
#    }
#
#ts=time.time()
#GASolutionEncoding = scp.geneticAlgorithmSetCovering(universe, subsets)
#execTime_GA = time.time()-ts
#scpResults["GA"]= {
#        "idx":scp.decodeSolution(GASolutionEncoding),
#        "enc":GASolutionEncoding,
#        "execTime":execTime_GA
#    }
#
#ts=time.time()
#GA_GreedySolutionEncoding = scp.geneticAlgorithmSetCovering(universe, subsets, greedySolutionEncoding)
#execTime_GA_Greedy = time.time()-ts
#scpResults["GA_Greedy"]= {
#        "idx":scp.decodeSolution(GA_GreedySolutionEncoding),
#        "enc":GA_GreedySolutionEncoding,
#        "execTime":execTime_GA_Greedy
#    }
#
#ts=time.time()
#CROSolutionEncoding = scp.chemicalReactionSetCovering(universe, subsets)
#execTime_CRO = time.time()-ts
#scpResults["CRO"]= {
#        "idx":scp.decodeSolution(CROSolutionEncoding),
#        "enc":CROSolutionEncoding,
#        "execTime":execTime_CRO
#    }
#
#ts=time.time()
#CRO_GreedySolutionEncoding = scp.chemicalReactionSetCovering(universe, subsets, greedySolutionEncoding)
#execTime_CRO_Greedy = time.time()-ts
#scpResults["CRO_Greedy"]= {
#        "idx":scp.decodeSolution(CRO_GreedySolutionEncoding),
#        "enc":CRO_GreedySolutionEncoding,
#        "execTime":execTime_CRO_Greedy
#    }

    
for res in scpResults:
    newKey = "rbPos_"+res
    dataCollection["robot"][newKey] = {}
    for i in scpResults[res]["idx"]:
        dataCollection["robot"][newKey][_allVPkeys_before[i]] = copy.deepcopy(dataCollection["robot"]["pos"][_allVPkeys_before[i]])
    
    resultFitness = scp.getFittness(universe, subsets, scpResults[res]["enc"])

    dataCollection["statistics"]["RbPos_setCovering"][res] = {}

    dataCollection["statistics"]["RbPos_setCovering"][res]["FitnessScore"] = resultFitness[0]
    dataCollection["statistics"]["RbPos_setCovering"][res]["Not Covered Count"] = resultFitness[1]
    dataCollection["statistics"]["RbPos_setCovering"][res]["Final ViewPoint Count"] = resultFitness[2]
    dataCollection["statistics"]["RbPos_setCovering"][res]["Unique Coverage Overlap (%)"] = resultFitness[3]
    dataCollection["statistics"]["RbPos_setCovering"][res]["Stacked Coverage Overlap (%)"] = resultFitness[4]
    dataCollection["statistics"]["RbPos_setCovering"][res]["Execution Time (s)"] = scpResults[res]["execTime"]
    

execTime = time.time()-ts
print("\n", "Time: ", execTime," seconds \n")

# Save statistics
dataCollection["statistics"]["execTimes"]["PHASE 4 STEP 1: Set Covering RbPos - TIME (s)"] = execTime



#______________________________________________________________________________________
# Finalizing an Saving results:
#______________________________________________________________________________________
print("\n \n \n============================================== \n",
    "Finalizing...")

ts=time.time()

# clean up for output (only setcovering VPS are relevant)
del dataCollection["robot"]["pos"]


now = datetime.now()
now_str = now.strftime("%y%m%d-%H%M%S")

filename = "00_Output Collection/PH4 output " + now_str +".json"

with open(filename, 'w') as f:
    json.dump(dataCollection, f, indent=2)


execTime = time.time()-ts
print("\n",
    "Time: ", execTime," seconds \n",
    "Results saved to file <", filename ,"> \n \n")
