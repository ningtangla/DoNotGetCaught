import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))

import random
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import itertools as it
import math
import pygame as pg
from pygame.color import THECOLORS

from src.visualization.drawDemo import DrawBackground, DrawState, VisualizeTraj, InterpolateStateForVisualization
from src.analyticGeometryFunctions import transCartesianToPolar, transPolarToCartesian
from src.MDPChasing.env import IsTerminal, IsLegalInitPositions, ResetState, PrepareSheepVelocity, PrepareWolfVelocity, PrepareDistractorVelocity, \
PrepareAllAgentsVelocities, StayInBoundaryByReflectVelocity, TransitWithInterpolation
from src.MDPChasing.reward import RewardFunctionTerminalPenalty
from src.MDPChasing.policies import RandomPolicy
from src.chooseFromDistribution import sampleFromDistribution, maxFromDistribution
from src.trajectory import ForwardOneStep, SampleTrajectory
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle

def composeFowardOneTimeStepWithRandomSubtlety(numOfAgent): # one time step used in different algorithems; here evaluate number of agent 
    # MDP 
    
    # experiment parameter for env
    numMDPTimeStepPerSecond = 5 #  change direction every 200ms 
    distanceToVisualDegreeRatio = 20

    minSheepSpeed = int(17.4 * distanceToVisualDegreeRatio/numMDPTimeStepPerSecond)
    maxSheepSpeed = int(23.2 * distanceToVisualDegreeRatio/numMDPTimeStepPerSecond)
    warmUpTimeSteps = 10 * numMDPTimeStepPerSecond # 10s to warm up
    prepareSheepVelocity = PrepareSheepVelocity(minSheepSpeed, maxSheepSpeed, warmUpTimeSteps)
    
    minWolfSpeed = int(8.7 * distanceToVisualDegreeRatio/numMDPTimeStepPerSecond)
    maxWolfSpeed = int(14.5 * distanceToVisualDegreeRatio/numMDPTimeStepPerSecond)
    wolfSubtleties = [500, 11, 3.3, 1.83, 0.92, 0.31, 0.001] # 0, 30, 60, .. 180
    initWolfSubtlety = np.random.choice(wolfSubtleties)
    prepareWolfVelocity = PrepareWolfVelocity(minWolfSpeed, maxWolfSpeed, warmUpTimeSteps, initWolfSubtlety, transCartesianToPolar, transPolarToCartesian)
    
    minDistractorSpeed = int(8.7 * distanceToVisualDegreeRatio/numMDPTimeStepPerSecond)
    maxDistractorSpeed = int(14.5 * distanceToVisualDegreeRatio/numMDPTimeStepPerSecond)
    prepareDistractorVelocity = PrepareDistractorVelocity(maxDistractorSpeed, maxDistractorSpeed, warmUpTimeSteps, transCartesianToPolar, transPolarToCartesian)
    
    sheepId = 0
    wolfId = 1
    distractorsIds = list(range(2, numOfAgent))
    prepareAllAgentsVelocities = PrepareAllAgentsVelocities(sheepId, wolfId, distractorsIds, prepareSheepVelocity, prepareWolfVelocity, prepareDistractorVelocity)

    xBoundary = [0, 600]
    yBoundary = [0, 600]
    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    
    killzoneRadius = 2.5 * distanceToVisualDegreeRatio
    isTerminal = IsTerminal(sheepId, wolfId, killzoneRadius)
 
    numFramePerSecond = 30 # visual display fps
    numFramesToInterpolate = int(numFramePerSecond / numMDPTimeStepPerSecond - 1) # interpolate each MDP timestep to multiple frames; check terminal for each frame

    transitFunction = TransitWithInterpolation(initWolfSubtlety, numFramesToInterpolate, prepareAllAgentsVelocities, stayInBoundaryByReflectVelocity, isTerminal)
    
    aliveBonus = 0.01
    deathPenalty = -1
    rewardFunction = RewardFunctionTerminalPenalty(aliveBonus, deathPenalty, isTerminal)

    forwardOneStep = ForwardOneStep(transitFunction, rewardFunction)

    return forwardOneStep
    
class SampleTrajectoriesForCoditions: # how to run episode/trajectory is different in alogrithems, here is a simple example to run episodes with fixed policy
    def __init__(self, numTrajectories, composeFowardOneTimeStepWithRandomSubtlety):
        self.numTrajectories = numTrajectories
        self.composeFowardOneTimeStepWithRandomSubtlety = composeFowardOneTimeStepWithRandomSubtlety
    
    def __call__(self, parameters):
        numOfAgent = parameters['numOfAgent']
        trajectories = []
        for trajectoryId in range(self.numTrajectories):
            
            forwardOneStep = self.composeFowardOneTimeStepWithRandomSubtlety(numOfAgent)
            
            sheepId = 0
            wolfId = 1
            distractorsIds = list(range(2, numOfAgent))
            distanceToVisualDegreeRatio = 20
            minInitSheepWolfDistance = 9 * distanceToVisualDegreeRatio
            minInitSheepDistractorDistance = 2.5 * distanceToVisualDegreeRatio  # no distractor in killzone when init
            isLegalInitPositions = IsLegalInitPositions(sheepId, wolfId, distractorsIds, minInitSheepWolfDistance, minInitSheepDistractorDistance)
            xBoundary = [0, 600]
            yBoundary = [0, 600]
            resetState = ResetState(xBoundary, yBoundary, numOfAgent, isLegalInitPositions, transPolarToCartesian)
            
            killzoneRadius = 2.5 * distanceToVisualDegreeRatio
            isTerminal = IsTerminal(sheepId, wolfId, killzoneRadius)
            
            numMDPTimeStepPerSecond = 5  
            maxRunningSteps = 25 * numMDPTimeStepPerSecond
            sampleTrajecoty = SampleTrajectory(maxRunningSteps, isTerminal, resetState, forwardOneStep)
            
            numActionDirections = 8
            actionSpace = [(np.cos(directionId * 2 * math.pi / numActionDirections), np.sin(directionId * 2 * math.pi / numActionDirections)) for directionId in range(numActionDirections)]
            randomPolicy = RandomPolicy(actionSpace)
            sampleAction = lambda state: sampleFromDistribution(randomPolicy(state))
             
            trajectory = sampleTrajecoty(sampleAction)

            trajectories.append(trajectory)
        return trajectories


def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numOfAgent'] = [2]
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]
 
    numTrajectories = 3
    sampleTrajectoriesForConditions = SampleTrajectoriesForCoditions(numTrajectories, composeFowardOneTimeStepWithRandomSubtlety)
    trajectoriesMultipleConditions = [sampleTrajectoriesForConditions(para) for para in parametersAllCondtion]

    visualConditionIndex = 0
    trajectoriesToVisualize = trajectoriesMultipleConditions[visualConditionIndex]

    visualize = True

    if visualize:
        screenWidth = 600
        screenHeight = 600
        screen = pg.display.set_mode((screenWidth, screenHeight))
        screenColor = THECOLORS['black']
        xBoundary = [0, 600]
        yBoundary = [0, 600]
        lineColor = THECOLORS['white']
        lineWidth = 4
        drawBackground = DrawBackground(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth)

        numOfAgent = 2
        numDistractors = numOfAgent - 2
        circleColorSpace = [[0, 255, 0], [255, 0, 0]] + [[255, 255, 255]] * numDistractors
        circleSize = 10
        positionIndex = [0, 1]
        agentIdsToDraw = list(range(numOfAgent))
        saveImage = False
        dirPYFile = os.path.dirname(__file__)
        imageSavePath = os.path.join(dirPYFile, '..', 'data', 'forDemo')
        if not os.path.exists(imageSavePath):
            os.makedirs(imageSavePath)
        FPS = 30
        drawState = DrawState(FPS, screen, circleColorSpace, circleSize, agentIdsToDraw, positionIndex,
                saveImage, imageSavePath, drawBackground)

       # MDP Env
        xBoundary = [0, 600]
        yBoundary = [0, 600]
        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
        
        distanceToVisualDegreeRatio = 20
        killzoneRadius = 2.5 * distanceToVisualDegreeRatio
        sheepId = 0
        wolfId = 1
        isTerminal = IsTerminal(sheepId, wolfId, killzoneRadius)
     
        numMDPTimeStepPerSecond = 5 #  change direction every 200ms 
        numFramesToInterpolate = int(FPS / numMDPTimeStepPerSecond - 1) # interpolate each MDP timestep to multiple frames; check terminal for each frame

        interpolateStateForVisualization = InterpolateStateForVisualization(numFramesToInterpolate, stayInBoundaryByReflectVelocity, isTerminal)

        stateIndexInTimeStep = 0
        actionIndexInTimeStep = 1
        nextStateIndexInTimeStep = 2
        visualizeTraj = VisualizeTraj(stateIndexInTimeStep, actionIndexInTimeStep, nextStateIndexInTimeStep, 
                drawState, interpolateStateForVisualization)

        [visualizeTraj(trajectory) for trajectory in trajectoriesToVisualize]


if __name__ == '__main__':
    main()


