import pygame as pg
import numpy as np
import os
import functools as ft

class DrawBackground:
    def __init__(self, screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth, xObstacles = None, yObstacles = None):
        self.screen = screen
        self.screenColor = screenColor
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.lineColor = lineColor
        self.lineWidth = lineWidth
        self.xObstacles = xObstacles
        self.yObstacles = yObstacles

    def __call__(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    exit()
        self.screen.fill(self.screenColor)
        rectPos = [self.xBoundary[0], self.yBoundary[0], self.xBoundary[1], self.yBoundary[1]]
        pg.draw.rect(self.screen, self.lineColor, rectPos, self.lineWidth)
        if self.xObstacles and self.yObstacles:
            for xObstacle, yObstacle in zip(self.xObstacles, self.yObstacles):
                rectPos = [xObstacle[0], yObstacle[0], xObstacle[1] - xObstacle[0], yObstacle[1] - yObstacle[0]]
                pg.draw.rect(self.screen, self.lineColor, rectPos)
        return

class DrawState:
    def __init__(self, fps, screen, colorSpace, circleSize, agentIdsToDraw, positionIndex, saveImage, imagePath, 
            drawBackGround):
        self.fps = fps
        self.screen = screen
        self.colorSpace = colorSpace
        self.circleSize = circleSize
        self.agentIdsToDraw = agentIdsToDraw
        self.xIndex, self.yIndex = positionIndex
        self.saveImage = saveImage
        self.imagePath = imagePath
        self.drawBackGround = drawBackGround

    def __call__(self, state):
        fpsClock = pg.time.Clock()
        
        self.drawBackGround()
        for agentIndex in self.agentIdsToDraw:
            positions, velocities = state
            agentPos = [np.int(positions[agentIndex][self.xIndex]), np.int(positions[agentIndex][self.yIndex])]
            agentColor = tuple(self.colorSpace[agentIndex])
            pg.draw.circle(self.screen, agentColor, agentPos, self.circleSize)

        pg.display.flip()
        
        if self.saveImage == True:
            filenameList = os.listdir(self.imagePath)
            pg.image.save(self.screen, self.imagePath + '/' + str(len(filenameList))+'.png')
        
        fpsClock.tick(self.fps)
        return self.screen

class InterpolateStateForVisualization:
    def __init__(self, numFramesToInterpolate, stayInBoundaryByReflectVelocity, isTerminal):
        self.numFramesToInterpolate = numFramesToInterpolate
        self.stayInBoundaryByReflectVelocity = stayInBoundaryByReflectVelocity
        self.isTerminal = isTerminal

    def __call__(self, state, action, nextState):
        currentAllAgentsPositions, lastAllAgentsVelocities = state
        nextAllAgentsPositions, currAllAgentsVelocities = nextState
        interpolatedStates = [state]
        for frameIndex in range(self.numFramesToInterpolate):
            noBoundaryNextPositions = np.array(currentAllAgentsPositions) + np.array(currAllAgentsVelocities)
            checkedNextPositionsAndVelocities = [self.stayInBoundaryByReflectVelocity(
                position, velocity) for position, velocity in zip(noBoundaryNextPositions, currAllAgentsVelocities)]
            nextAllPositionsForInterpolation, nextAllVelocitiesForInterpolation = list(zip(*checkedNextPositionsAndVelocities))
            nextStateForVisualization = np.array([nextAllPositionsForInterpolation, nextAllVelocitiesForInterpolation])
            interpolatedStates.append(nextStateForVisualization)
            currentAllAgentsPositions = nextAllPositionsForInterpolation
            currAllAgentsVelocities = nextAllVelocitiesForInterpolation
            if self.isTerminal(nextStateForVisualization):
                break
        return interpolatedStates

class VisualizeTraj:
    def __init__(self, stateIndex, actionIndex, nextStateIndex, drawState, interpolateState = None):
        self.stateIndex = stateIndex
        self.actionIndex = actionIndex
        self.nextStateIndex = nextStateIndex
        self.drawState = drawState
        self.interpolateState = interpolateState

    def __call__(self, trajectory):
        for timeStepIndex in range(len(trajectory)):
            timeStep = trajectory[timeStepIndex]
            state = timeStep[self.stateIndex]
            action = timeStep[self.actionIndex]
            nextState = timeStep[self.nextStateIndex]
            if self.interpolateState and timeStepIndex!= len(trajectory) - 1:
                statesToDraw = self.interpolateState(state, action, nextState)
            else:
                statesToDraw  = [state]
            for state in statesToDraw:
                screen = self.drawState(state)
        return
