import copy
import random
import math
import numpy as np
import itertools
import matplotlib.pyplot as plt 


#######################################################################################
#######################################################################################
# HELPER FUNCTIONS
def encodeSolution(idxList, subsets):
    binaryEncoding = [0]*len(subsets)
    for _idx in idxList:
        binaryEncoding[_idx] = 1
    return binaryEncoding

def decodeSolution(binaryEncoding):
    idxList = list()
    for setIter, value in enumerate(binaryEncoding):
        if(value != 0):
            idxList.append(setIter)
    return idxList

def createRandomSolution(universe, subsets):
    numSets = len(subsets)

    randSol = [0]*numSets
    covered = set()

    while covered != universe:
        _idx = np.random.randint(0, numSets-1)
        _coverage = subsets[_idx] - covered
        if len(_coverage) > 0:
            randSol[_idx]=1      
            covered |= _coverage
    
    return randSol

def getNeighborSolution(_solution, maxMutAmount, maxChangeDist):
    #select how many to mutate
    mutations = np.random.randint(1,maxMutAmount)
    usedIndices = np.argwhere(_solution).reshape(-1).tolist()

    if(mutations > len(usedIndices)):
        mutations = len(usedIndices)

    idxToChange = np.random.choice(usedIndices, int(mutations), replace=False) # choose random indices as candidates

    newSol = copy.deepcopy(_solution)
    for i_mu in idxToChange:
        signIndicator = np.random.randint(0,1)
        changeDist = np.random.randint(1,maxChangeDist)

        newSol[i_mu]=0

        newIdx = 0
        if (signIndicator==1):
            if i_mu+changeDist > len(newSol):
                newIdx = i_mu-changeDist
            else:
                newIdx = i_mu+changeDist
        else:
            if i_mu-changeDist < 0:
                newIdx = i_mu+changeDist
            else:
                newIdx = i_mu-changeDist

        newSol[newIdx] = 1

    return newSol

def cleanUpSolution(universe, subsets, _solution):
    # clean up Chromo 
    newCoverage = set()
    cleanedSol = [0]*len(subsets)
    for i_sol, value in enumerate(_solution):
        if newCoverage == universe:
            break
        if value == 1:
            _coverage = subsets[i_sol] - newCoverage
            if len(_coverage) > 0:
                newCoverage |= subsets[i_sol]
                cleanedSol[i_sol] = 1
    
    # complete the solution with greedy approach
    while newCoverage != universe:
        subset = max(subsets, key=lambda s: len(s - newCoverage))
        cleanedSol[subsets.index(subset)] = 1 
        newCoverage |= subset

    return cleanedSol

def getFittness(universe, subsets, _solution):
    _universe_len = len(universe)
    numSetsUsed = 0
    covered = set()
    stacked_overlap = list()
    unique_overlap = set()

    notCovered_weight = 1000
    numSetsUsed_weight = 1
    unique_overlap_weight = 1
    stacked_overlap_weight = 1

    for setIter, value in enumerate(_solution):
        if(value != 0):
            numSetsUsed +=1
            cur_newCovered = subsets[setIter] - covered
            cur_alreadyCovered = subsets[setIter] & covered
            covered |= cur_newCovered
            stacked_overlap.extend(list(cur_alreadyCovered))
            unique_overlap |= cur_alreadyCovered
    
    out_notCovered = (1-(len(covered)/_universe_len))
    out_numSetsUsed = numSetsUsed
    out_unique_overlap = (len(unique_overlap)/_universe_len)
    out_stacked_overlap = (len(stacked_overlap)/_universe_len)

    out_fitness = notCovered_weight * out_notCovered + numSetsUsed_weight * out_numSetsUsed + unique_overlap_weight * out_unique_overlap + stacked_overlap_weight * out_stacked_overlap

    return out_fitness, out_notCovered, out_numSetsUsed, out_unique_overlap, out_stacked_overlap

def generateSplitPoint(Sets):
    a = np.random.randint(0,len(Sets))
    b = a
    while(a == b):
        b = np.random.randint(0,len(Sets))   

    splitPoint1 = min(a, b)
    splitPoint2 = max(a, b) 
    return splitPoint1, splitPoint2

def Crossover(parent1genes, parent2genes, splitPoint1, splitPoint2):
    tempChromosome = []
    #First section
    tempChromosome = tempChromosome + (list(parent1genes[0:splitPoint1]))

    #Mid Section
    tempChromosome = tempChromosome + (list(parent2genes[splitPoint1:splitPoint2]))

    #Last Section    
    tempChromosome = tempChromosome + (list(parent1genes[splitPoint2:]))

    return tempChromosome  

#######################################################################################
#######################################################################################
# MAIN FUNCTIONS
def greedySetCovering(universe, subsets):
    #Find a family of subsets that covers the universal set
    covered = set()
    subset_id = []
    # Greedily add the subsets with the most uncovered points
    while covered != universe:
        subset = max(subsets, key=lambda s: len(s - covered))
        subset_id.append(subsets.index(subset))
        covered |= subset
 
    return subset_id

def naiveSetCovering(universe, subsets):
    covered = set()
    subsets_ids_ordered = list(range(0, len(subsets)))
    result_ids = []

    def getElemLen(elem):
        return len(subsets[elem])

    subsets_ids_ordered.sort(key=getElemLen, reverse=True)
    i = 0
    while covered != universe or i<=len(subsets)-1:
        _curSet = subsets[subsets_ids_ordered[i]]
        if  len(_curSet - covered) > 0:
            result_ids.append(subsets_ids_ordered[i])
            covered |= _curSet
        i+=1
    
    return result_ids

def simulatedAnnealingSetCovering(universe, subsets):
    # code based on: https://dev.to/cesarwbr/how-to-implement-simulated-annealing-algorithm-in-python-4gid
    # see also: https://machinelearningmastery.com/simulated-annealing-from-scratch-in-python/

    """Peforms simulated annealing to find a solution"""
    initial_temp = 90
    final_temp = .1
    alpha = 0.01
    

    current_temp = initial_temp

    # Start by initializing the current state with the initial state
    current_state = createRandomSolution(universe, subsets)
    solution = current_state

    iterator = 0

    solutionSet = []
    temperatureSet = []


    while current_temp > final_temp:
        
        # Check if neighbor is best so far
        fitness_s = getFittness(universe, subsets, solution)

        neighbor = createRandomSolution(universe, subsets)
        neighbor = cleanUpSolution(universe, subsets, neighbor)

        fitness_n = getFittness(universe, subsets, neighbor)

        cost_diff =  fitness_s[0] - fitness_n[0]
        #print(fitness_s, " Temp: ", current_temp)

        # if the new solution is better, accept it
        if cost_diff > 0:
            solution = neighbor
        # if the new solution is not better, accept it with a probability of e^(-cost/temp)
        else:
            if random.uniform(0, 1) < math.exp(cost_diff / current_temp):
                solution = neighbor
        
        # save data for visualization
        #solutionSet.append(fitness_s[0])
        #temperatureSet.append(current_temp)

        # decrement the temperature
        #current_temp -= alpha

        #Fast SA
        current_temp = initial_temp / (iterator+1)
        iterator+=1
    

    # optional visualize
    # x axis values 
    #y = solutionSet 
    ## corresponding y axis values 
    #x = temperatureSet
#
    ## plotting the points  
    #plt.plot(x, y) 
#
    ## naming the x axis 
    #plt.xlabel('x - axis') 
    ## naming the y axis 
    #plt.ylabel('y - axis') 
#
    ## giving a title to my graph 
    #plt.title('My first graph!') 
#
    ## function to show the plot 
    #plt.show() 
    
    return solution

def geneticAlgorithmSetCovering(universe, subsets, greedySolutionEncoding=None):
    #code based on: https://github.com/BurkhardtMicah/Artificial-Intelligence/blob/main/HybridGA-WOC_SetCoverProblem.py
    class chromosome():
        #A binary representation that shows whether a gene is included or not along with how fit the solution is
        def __init__(self, geneList):
            self.genes = geneList
            self.fitness = math.inf
    def sortChromosomes(Population):
        outputPopArr = []
        for chromo in Population:
            dist = chromo.fitness
            if(len(outputPopArr) == 0):
                outputPopArr.append(chromo)
            else:
                i = 0
                added = False
                for outChromo in outputPopArr:
                    if(outChromo.fitness >= dist):
                        outputPopArr.insert(i, chromo)
                        added = True
                        break
                    else:
                        #if chromosome is larger than all current, place at end
                        i += 1
                if(not added):
                    outputPopArr.append(chromo)
        return outputPopArr
    def getNumMating(gaPopSize, matingPercentage):
        numMatingChromosomes = int(matingPercentage * gaPopSize)
        if(numMatingChromosomes % 2 == 1 and numMatingChromosomes < gaPopSize):
            numMatingChromosomes += 1
        elif(numMatingChromosomes % 2 == 1 and numMatingChromosomes >= gaPopSize):
            numMatingChromosomes -= 1
        elif(numMatingChromosomes == 0):
            numMatingChromosomes = 2   
        return numMatingChromosomes #else even pairs - do nothing  
    def pairMates(matingChromosomes, numMatingChromosomes):
        used        = []
        pairedMates = []

        i = 1
        j = numMatingChromosomes - 1
        if(i == j):
            topPerformersMate = 1
        else:
            topPerformersMate = np.random.randint(1,j)  
        for pair in itertools.combinations(matingChromosomes,2):
            c1 = pair[0]
            c2 = pair[1]

            if(i == topPerformersMate):
                used.append(c1)
                used.append(c2) 
                pairedMates.append(pair)
                i = i+1
                continue
            elif(not(used.__contains__(c1) or used.__contains__(c2)) and i > topPerformersMate):
                used.append(c1)
                used.append(c2)
                pairedMates.append(pair)
            i = i+1   
        return pairedMates
    
    NumGAPopulations = 1
    gaPopSize        = 1000
    numNoChangeGen   = 25
    i_max            = 500
    mutationChance   = .3 #Percent Chance of motation
    geneMutAmount    = 10
    matingPercentage = .7 #Percent of chromosomes to mate

    fittestSolution = [math.inf, math.inf]
    prevSolFitness = math.inf
    resultCollection={}

    for pop in range(NumGAPopulations):

        _popKey = "pop_" + str(pop)
        resultCollection[_popKey]={}
        
        #print("Population: " + str(pop))

        ############
        #INITIALIZE
        ###########

        #Create population of chromosomes - binary lists
        Population = []

        #create array of nodes
        for i in range(gaPopSize):
            Population.append(chromosome(createRandomSolution(universe, subsets)))
        
        # Optional: Add Greedy solution to the population
        if(greedySolutionEncoding != None):
            Population.append(chromosome(greedySolutionEncoding))
            gaPopSize+=1

        
        ###########################################################################
        genNumber = 0
        noChange = 0

        #while(numNoChangeGen > noChange):
        while(genNumber < i_max):

            #################################
            #TEST FITNESS OF EACH CHROMOSOME
            ################################

            for i in (range(len(Population))):
                Population[i].fitness = getFittness(universe, subsets, Population[i].genes)[0]

            #############################
            #SELECT FITTEST CHROMOSOMES
            ###########################

            #get number of chromosomes to mate
            numMatingChromosomes = getNumMating(gaPopSize, matingPercentage)

            #sort based on fittness -AKA smallest distance is first
            Population = sortChromosomes(Population)

            #Find and select n fittest chromosomes - select the numMatingChromosomes number of chromosomes that are fittest
            matingChromosomes = Population[0:numMatingChromosomes]

            #Pair mates somewhat randomly
            pairedMates = pairMates(matingChromosomes, numMatingChromosomes)


            ##################################
            #Graph Current Fittest chromosome -Graphs best fitness on each iteration with change
            #################################
            curBestSolFitness = Population[0].fitness

            #check if a new optimum is found
            if(prevSolFitness != curBestSolFitness):
                if fittestSolution[0] > curBestSolFitness:
                    fittestSolution[0] = curBestSolFitness
                    fittestSolution[1] = Population[0].genes

                noChange = 0
            else:
                noChange +=1

            prevSolFitness = curBestSolFitness

            #######################
            #CROSSOVER - TWO POINT
            ######################

            #Split chromosome at split Point
            tempChromosome1 = []
            tempChromosome2 = []
            i = 0
            for mates in pairedMates:
                #index of chromosome to be split at for crossover
                splitPoint1, splitPoint2 = generateSplitPoint(subsets)

                #Get parent genes
                parent1genes = mates[0].genes
                parent2genes = mates[1].genes

                #pull first n nodes from parent1 til min split point
                tempChromosome1 = Crossover(parent1genes, parent2genes, splitPoint1, splitPoint2)
                tempChromosome2 = Crossover(parent2genes, parent1genes, splitPoint1, splitPoint2)

                newChromos = [tempChromosome1, tempChromosome2]

                ###########
                #MUTATION
                #########
                #1 in mutation chance: chance of mutation occurring on gene
                for newCIdx, chromo in enumerate(newChromos):

                    if(random.random() < mutationChance): #Mutation Occurs
                        #select how many genes to mutate
                        mutations = np.random.randint(1,geneMutAmount)

                        for _i in range(mutations):
                            mutationIndex = np.random.randint(0,(len(chromo)))
                            if(chromo[mutationIndex] == 0):
                                chromo[mutationIndex] = 1
                            else:
                                chromo[mutationIndex] = 0

                    
                    # clean up Chromo 
                    newCleanedChromo = cleanUpSolution(universe, subsets, chromo)
                                        
                    #replace genes in worst performers in initial population
                    Population[(gaPopSize - i - (newCIdx + 1))].genes = newCleanedChromo[:]       
               
                i += 2  #increment by two since we are doing two genes per iteration      
           

            _genKey = "gen_" + str(genNumber)
            #a= sum(s[0] for s in  Population)
            resultCollection[_popKey][_genKey]=(genNumber,Population[0].fitness, sum(s.fitness for s in Population) / len(Population))

            genNumber += 1
    
    # optional visualize
    #x axis values 
    #y1 = list(resultCollection['pop_0'][s][1] for s in resultCollection['pop_0']) 
    #y2 = list(resultCollection['pop_0'][s][2] for s in resultCollection['pop_0']) 
#
    ## corresponding y axis values 
    #x = list(resultCollection['pop_0'][s][0] for s in resultCollection['pop_0'])
#
    ## plotting the points  
    #plt.plot(x, y1)  
    #plt.plot(x, y2)  
#
#
    ## naming the x axis 
    #plt.xlabel('x - axis') 
    ## naming the y axis 
    #plt.ylabel('y - axis') 
#
    ## giving a title to my graph 
    #plt.title('My first graph!') 
#
    ## function to show the plot 
    #plt.show() 

    return fittestSolution[1]

def chemicalReactionSetCovering(universe, subsets,greedySolutionEncoding=None):
    # code based on: https://github.com/sha443/Chemical-Reaction-Optimization-CRO

    # Parameters
    maxCROiterations = 4000 #10000
    CROpopSize = 10 #100 #500
    KELossRate= 0.2 #0.1#0.85
    MoleculeCollisionChance= 0.2 #0.50
    InitialKE= 1000# 100
    alpha = 500 #500 #250 #1
    beta = 10
    global EnrgBuffer
    EnrgBuffer = 0 #1000
    MoleculeMutAmount = 10
    maxStructureChangeDist = 5
    minEnrg = math.inf
    minEnrgIdx = None
    minEnrgStructure = None

    resultCollection={}

    # Oprators hit counters
    global on, dec, inef, syn
    on = 0
    dec = 0
    inef = 0
    syn = 0

    class Molecule():
        def __init__(self, structure, initialKE):
            self.structure = structure
            self.PE = getFittness(universe, subsets, self.structure)[0]
            self.KE = initialKE
            self.numHit = 0

            #local known minimum
            self.minHit = 0
            self.minPE = self.PE
            self.minStruct = self.structure

    def OnwallIneffectiveCollision(molIndex):
        global EnrgBuffer, on
        newMol = getNeighborSolution(reagentTray[molIndex].structure, MoleculeMutAmount, maxStructureChangeDist)
        newMol = cleanUpSolution(universe, subsets, newMol)

        PEnew = getFittness(universe, subsets, newMol)[0]  
        KEnew = 0.0
        reagentTray[molIndex].numHit += 1

        t = reagentTray[molIndex].PE + reagentTray[molIndex].KE 
        if(t>=PEnew):

            on += 1


            a = (random.uniform(0,1) * (1-KELossRate))+KELossRate
            KEnew = (reagentTray[molIndex].PE - PEnew + reagentTray[molIndex].KE)*a
            EnrgBuffer = EnrgBuffer + (reagentTray[molIndex].PE - PEnew + reagentTray[molIndex].KE) * (1 - a)
            reagentTray[molIndex].structure = newMol
            reagentTray[molIndex].PE = PEnew
            reagentTray[molIndex].KE = KEnew

            #Update local known minimum
            if(reagentTray[molIndex].PE < reagentTray[molIndex].minPE):
                reagentTray[molIndex].minStruct = reagentTray[molIndex].structure
                reagentTray[molIndex].minPE = reagentTray[molIndex].PE
                reagentTray[molIndex].minHit = reagentTray[molIndex].numHit

    def Decomposition(molIndex):
        global EnrgBuffer, dec
        molecule = reagentTray[molIndex].structure
        
        #Perform decomposition
        #----------------------------------------------------------------------------------------------
        length = len(molecule)
        m1 = [0]*length
        m2 = [0]*length
        mid =int(length/2)


        # First half goes to the first half of the new molecule1
        for i in range(0,mid):
            m1[i] = molecule[i]
        #Endfor
        # Second half goes to the second half of the new molecule2
        for i in range(mid,length):
            m2[i] = molecule[i]
        #Endfor

        # Molecule1 second half randomly chosen
        for i in range(mid,length):
            m1[i] = random.randint(0, 1)
        #Endfor

        # Molecule2 first half randomly chosen
        for i in range(0,mid):
            m2[i] = random.randint(0, 1)
        #Endfor

        #handle decomposition result
        #----------------------------------------------------------------------------------------------
        newMol1 = cleanUpSolution(universe, subsets, m1)
        newMol2 = cleanUpSolution(universe, subsets, m2)


        pe1 = getFittness(universe, subsets, newMol1)[0]  
        pe2 = getFittness(universe, subsets, newMol2)[0]

        t = reagentTray[molIndex].PE + reagentTray[molIndex].KE

        p = random.random()
        p1 = random.random() / 10
        p2 = random.random() 
        p3 = random.random()
        #p4 = random.random()

        if (t >= (pe1 + pe2)): # energy already is enough for the reaction
            
            dec += 1


            ke1 = (t - (pe1 + pe2)) * p
            ke2 = (t - (pe1 + pe2)) * (1 - p)
            EnrgBuffer = EnrgBuffer + (t - (pe1 + pe2)) - ke1 - ke2

            # replace initial Molecule with the first new one
            reagentTray[molIndex].structure = newMol1
            reagentTray[molIndex].PE = pe1
            reagentTray[molIndex].KE = ke1
            reagentTray[molIndex].numHit = 0
            reagentTray[molIndex].minHit = 0
            reagentTray[molIndex].minStruct = newMol1
            reagentTray[molIndex].minPE = pe1

            # add new molecule to the population
            reagentTray.append(Molecule(newMol2, ke2))

        elif ((t + (EnrgBuffer * p1 * p2)) >= (pe1 + pe2)): # energy including the buffer is enough for the reaction

            dec += 1

            e_decrease = (t + (EnrgBuffer * p1 * p2)) - (pe1 + pe2)
            ke1 = e_decrease * p3
            ke2 = e_decrease * (1 - p3)
            EnrgBuffer = (1 - p1 * p2) * EnrgBuffer

        


            #ke1 = ((t - pe1 - pe2) + EnrgBuffer) * (p1*p2)
            #ke2 = ((t - pe1 - pe2) + EnrgBuffer) * (p3*p4)
            #EnrgBuffer = EnrgBuffer + (t - (pe1 + pe2)) - ke1 - ke2

            # replace initial Molecule with the first new one
            reagentTray[molIndex].structure = newMol1
            reagentTray[molIndex].PE = pe1
            reagentTray[molIndex].KE = ke1
            reagentTray[molIndex].numHit = 0
            reagentTray[molIndex].minHit = 0
            reagentTray[molIndex].minStruct = newMol1
            reagentTray[molIndex].minPE = pe1

            # add new molecule to the population
            reagentTray.append(Molecule(newMol2, ke2))

        else: # reaction does not happen
            reagentTray[molIndex].numHit += 1  
        
    def Synthesis(molIndex1, molIndex2):
        global syn
        molecule1 = reagentTray[molIndex1].structure
        molecule2 = reagentTray[molIndex2].structure

        #Perform synthesis (recombination similar to the crossover in GA)
        #----------------------------------------------------------------------------------------------
        #index of chromosome to be split at for crossover
        splitPoint1, splitPoint2 = generateSplitPoint(subsets)

        #Get parent genes
        parent1genes = molecule1
        parent2genes = molecule2

        #pull first n nodes from parent1 til min split point
        m = Crossover(parent1genes, parent2genes, splitPoint1, splitPoint2)

        # Alternative do random Crossover
        #length = len(molecule1)
        #m = list(range(length))
        #for i in range(0,length):
        #    r = random.uniform(0, 1)
        #    if (r<.5):
        #        m[i] = molecule1[i]
        #    else:
        #        m[i] = molecule2[i]

        #handle synthesis result
        #----------------------------------------------------------------------------------------------
        newMol = cleanUpSolution(universe, subsets, m)

        PEnew = getFittness(universe, subsets, newMol)[0]

        t1 = reagentTray[molIndex1].PE + reagentTray[molIndex1].KE
        t2 = reagentTray[molIndex2].PE + reagentTray[molIndex2].KE

        if((t1 + t2) >= PEnew): # reaction will happen

            syn += 1

            KEnew = (t1 + t2) - PEnew

            # delete the two molecules the had the reaction
            del reagentTray[molIndex1]
            if(molIndex2 >= molIndex1):
				# position of index2 is decreased by 1
                molIndex2 -= 1
            del reagentTray[molIndex2]

            # add new Molecute to the population
            reagentTray.append(Molecule(newMol, KEnew))

        else: # reaction does not happen
            reagentTray[molIndex1].numHit += 1
            reagentTray[molIndex2].numHit += 1

    def IntermolecularIneffectiveCollision(molIndex1, molIndex2):
        global inef
        molecule1 = reagentTray[molIndex1].structure
        molecule2 = reagentTray[molIndex2].structure

        # perform inter 
        #----------------------------------------------------------------------------------------------
        length1 = len(molecule1)
        length2 = len(molecule2)
        m1 = list(range(length1))
        m2 = list(range(length2))

        #Random numbers x1, x2 generation
        x1 = random.randint(0, length1-1)
        x2 = random.randint(0, length2-1)

        # Randormly choose form molecule1 or molecule2
        for i in range(0,length1):
            if (i<x1 or i>x2):  #if odd segments
                m1[i] = molecule1[i]
                m2[i] = molecule2[i]
            elif (i>=x1 and x2>=i): # if even segment
                m1[i] = molecule2[i]
                m2[i] = molecule1[i]

        #handle inter result
        #----------------------------------------------------------------------------------------------
        newMol1 = cleanUpSolution(universe, subsets, m1)
        newMol2 = cleanUpSolution(universe, subsets, m2)

        pe1 = getFittness(universe, subsets, newMol1)[0]  
        pe2 = getFittness(universe, subsets, newMol2)[0]

        t1 = reagentTray[molIndex1].PE + reagentTray[molIndex1].KE
        t2 = reagentTray[molIndex2].PE + reagentTray[molIndex2].KE

        e_inter = (t1 +t2) - (pe1 + pe2)
        gamma4 = random.uniform(0,1)

        reagentTray[molIndex1].numHit += 1
        reagentTray[molIndex2].numHit += 1

        if(e_inter >= 0): # reaction will happen

            inef += 1


            # replace Moleculewith the new ones
            reagentTray[molIndex1].structure = newMol1
            reagentTray[molIndex1].PE = pe1
            reagentTray[molIndex1].KE = e_inter * gamma4

            reagentTray[molIndex2].structure = newMol2
            reagentTray[molIndex2].PE = pe2
            reagentTray[molIndex2].KE = e_inter * (1 - gamma4)

            # update local known minimum
            if (reagentTray[molIndex1].PE < reagentTray[molIndex1].minPE):
                reagentTray[molIndex1].minStruct = reagentTray[molIndex1].structure
                reagentTray[molIndex1].minPE = reagentTray[molIndex1].PE
                reagentTray[molIndex1].minHit = reagentTray[molIndex1].numHit
            
            if (reagentTray[molIndex2].PE < reagentTray[molIndex2].minPE):
                reagentTray[molIndex2].minStruct = reagentTray[molIndex2].structure
                reagentTray[molIndex2].minPE = reagentTray[molIndex2].PE
                reagentTray[molIndex2].minHit = reagentTray[molIndex2].numHit
 

    #----------------------------------------------------------------------------------------------
    # Population generation
    #----------------------------------------------------------------------------------------------
    reagentTray = []
    for i_ps in range(CROpopSize):
        reagentTray.append(Molecule(createRandomSolution(universe, subsets), InitialKE))
    
    # Optional: Add Greedy solution to the population
    if(greedySolutionEncoding != None):
        reagentTray.append(Molecule(greedySolutionEncoding,InitialKE))
        CROpopSize+=1

    #----------------------------------------------------------------------------------------------
	# Optimize with CRO
	#----------------------------------------------------------------------------------------------
    for i_cro in range(maxCROiterations):
        if(random.random()>MoleculeCollisionChance or len(reagentTray)==1): # Unimolecular or intermolecular 
            # Unimolecular
            randMolIdx = random.randint(0, len(reagentTray)-1)
            # Decomposition or OnwallIneffectiveCollision
            if(reagentTray[randMolIdx].numHit - reagentTray[randMolIdx].minHit > alpha):
                # Decomposition
                Decomposition(randMolIdx)
                #dec += 1
            else:
                # OnwallIneffectiveCollision
                OnwallIneffectiveCollision(randMolIdx)
                #on += 1
        else:
            # intermolecular
            molIndices = range(len(reagentTray))
            ch = np.random.choice(molIndices, 2, replace=False)
            randMolIdx1 = ch[0]
            randMolIdx2 = ch[1]

            if((reagentTray[randMolIdx1].KE + reagentTray[randMolIdx2].KE) < beta):# Synthesis or IntermolecularIneffectiveCollision
                # Synthesis
                Synthesis(randMolIdx1, randMolIdx2)
                #syn += 1
            else:
                # IntermolecularIneffectiveCollision
                IntermolecularIneffectiveCollision(randMolIdx1, randMolIdx2)
                #inef += 1

        # Finding minimum energy
        sumPE = 0
        sumKE = 0
        for i_rt, ithMol in enumerate(reagentTray):
            sumPE += ithMol.PE
            sumKE += ithMol.KE
            if(ithMol.minPE < minEnrg):
                minEnrg = ithMol.minPE
                minEnrgIdx = i_rt
                minEnrgStructure = reagentTray[minEnrgIdx].minStruct
        
        sumE = sumPE+sumKE+EnrgBuffer

        _genKey = "corIter_" + str(i_cro)
        #a= sum(s[0] for s in  Population)
        resultCollection[_genKey]=(i_cro,minEnrg, sum(s.PE for s in reagentTray) / len(reagentTray), sumE, sumPE,sumKE,EnrgBuffer, dec, on, syn, inef, len(reagentTray), sum(s.numHit for s in reagentTray) / len(reagentTray))
    
    
    ## optional visualize
    #fig, axs = plt.subplots(2, sharex=True)
    ##x axis values 
    #y1 = list(resultCollection[s][1] for s in resultCollection) 
    #y2 = list(resultCollection[s][2] for s in resultCollection)
    #y3 = list(resultCollection[s][3] for s in resultCollection) 
    #y4 = list(resultCollection[s][4] for s in resultCollection)    
    #y5 = list(resultCollection[s][5] for s in resultCollection) 
    #y6 = list(resultCollection[s][6] for s in resultCollection)
    #y7 = list(resultCollection[s][7] for s in resultCollection) 
    #y8 = list(resultCollection[s][8] for s in resultCollection)    
    #y9 = list(resultCollection[s][9] for s in resultCollection) 
    #y10 = list(resultCollection[s][10] for s in resultCollection)
    #y11 = list(resultCollection[s][11] for s in resultCollection) 
    #y12 = list(resultCollection[s][12] for s in resultCollection)
#
    ## corresponding y axis values 
    #x = list(resultCollection[s][0] for s in resultCollection)
#
    ## plotting the points  
    #axs[0].plot(x, y1, label = "best F")  
    #axs[0].plot(x, y2, label = "avgF")  
    #
    #axs[1].plot(x, y3, label = "sumE")  
    #axs[1].plot(x, y4, label = "sumPE")   
    #axs[1].plot(x, y5, label = "sumKE")  
    #axs[1].plot(x, y6, label = "buffer")  

    #axs[2].plot(x, y7, label = "dec")  
    #axs[2].plot(x, y8, label = "on")   
    #axs[2].plot(x, y9, label = "syn")  
    #axs[2].plot(x, y10, label = "inef") 
#
    #axs[3].plot(x, y11, label = "popSize")  
    #axs[4].plot(x, y12, label = "avgHit")


    # function to show the plot 
    #axs[0].legend()
    #axs[1].legend()
   ## axs[2].legend()
   ## axs[3].legend()
   ## axs[4].legend()
#
    #fig.show() 
#
#
#
    #hits = "Buffer: "+str(EnrgBuffer)+"\tOnwall= "+str(on) +"\tDec = "+str(dec)+"\tSyn = "+str(syn)+"\tIntermolecular = "+str(inef)+"\n"
	#
	## Logs:
    #print(hits, minEnrg)
    
    return minEnrgStructure

def particleSwarmSetCovering(universe, subsets,greedySolutionEncoding=None):
    PSOpopSize = 50
    maxPSOiterations = 500
    resultCollection={}

    class Particle():
        def __init__(self, structure):
            self.structure = structure
            self.fitness = getFittness(universe, subsets, self.structure)[0]
            self.strIdx = np.argwhere(self.structure).reshape(-1).tolist()
            
            # local known minimum
            self.ownBestStruct = self.structure
            self.ownBestFitness = self.fitness
            self.ownBestStrIdx = self.strIdx

    PSOglobalBest = Particle([0])

    #----------------------------------------------------------------------------------------------
    # Population generation
    #----------------------------------------------------------------------------------------------
    swarm = []
    for i_ps in range(PSOpopSize):
        swarm.append(Particle(createRandomSolution(universe, subsets)))
    
    # Optional: Add Greedy solution to the population
    if(greedySolutionEncoding != None):
        swarm.append(Particle(greedySolutionEncoding))
        PSOpopSize+=1

    # get initial best solution
    for i_sw, ithpart in enumerate(swarm):
        if(ithpart.fitness < PSOglobalBest.fitness):
            PSOglobalBest = ithpart

    #----------------------------------------------------------------------------------------------
	# Optimize with PSO
	#----------------------------------------------------------------------------------------------
    for i_pso in range(maxPSOiterations):
        for i_partIdx, ithPart in enumerate(swarm):
            attractorDecision = random.randint(0,3)
            if(attractorDecision == 0):
                # move towards global best solution
                attractor = PSOglobalBest.strIdx
            elif (attractorDecision == 1):
                # move towards a random solution
                attractor = np.argwhere(createRandomSolution(universe, subsets)).reshape(-1).tolist()
            elif (attractorDecision == 2):
                # move towards best neighbor solution
                n1_idx = i_partIdx-1
                if n1_idx < 0:
                    n1_idx = 2
                n2_idx = i_partIdx+1
                if n2_idx >= len(swarm):
                    n2_idx = len(swarm)-3

                n_bef = swarm[n1_idx].fitness
                n_aft = swarm[n2_idx].fitness

                if n_bef <= n_aft:
                    attractor = swarm[n1_idx].strIdx
                else:
                    attractor = swarm[n2_idx].strIdx
            else: #attractorDecision == 3
                # move towards own best solution
                attractor = ithPart.strIdx
            

            # merge current solution with the attractor
            usedIndicesSelf = ithPart.strIdx
            usedIndicesAttractor = attractor

            mutationAmount = random.randint(0, 10)#len(usedIndicesSelf))
            newParticle = copy.deepcopy(ithPart)

            for i_ma in range(mutationAmount):
                mergeDecision = random.random()
                if mergeDecision <= 0.5:
                    randIDx = np.random.choice(usedIndicesSelf, 1)[0]
                    newParticle.structure[randIDx] = 0
                else:
                    randIDx = np.random.choice(usedIndicesAttractor, 1)[0]
                    newParticle.structure[randIDx] = 1

            newParticle.structure = cleanUpSolution(universe, subsets, newParticle.structure)
            newFitness = getFittness(universe, subsets, newParticle.structure)[0]

            if newFitness < ithPart.fitness:
                ithPart.structure = newParticle.structure
                ithPart.fitness = newFitness
                ithPart.strIdx = np.argwhere(newParticle.structure).reshape(-1).tolist()
            if newFitness < PSOglobalBest.fitness:
                PSOglobalBest = newParticle
            if newFitness < ithPart.ownBestFitness:
                ithPart.ownBestStruct = newParticle.structure
                ithPart.ownBestFitness = newFitness
                ithPart.ownBestStrIdx = np.argwhere(newParticle.structure).reshape(-1).tolist()


        _genKey = "iter_" + str(i_pso)
        #a= sum(s[0] for s in  Population)
        resultCollection[_genKey]=(i_pso,PSOglobalBest.fitness, sum(s.fitness for s in swarm) / len(swarm))
    
    
    ## optional visualize
    ##x axis values 
    #y1 = list(resultCollection[s][1] for s in resultCollection) 
    #y2 = list(resultCollection[s][2] for s in resultCollection) 
#
    ## corresponding y axis values 
    #x = list(resultCollection[s][0] for s in resultCollection)
#
    ## plotting the points  
    #plt.plot(x, y1)  
    #plt.plot(x, y2)  
#
#
    ## naming the x axis 
    #plt.xlabel('x - axis') 
    ## naming the y axis 
    #plt.ylabel('y - axis') 
#
    ## giving a title to my graph 
    #plt.title('My first graph!') 
#
    ## function to show the plot 
    #plt.show() 


    return PSOglobalBest.structure


