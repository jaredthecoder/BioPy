import numpy as np
import random
from Problems import *
from Particle import *
class Swarm:
	def __init__(npart = 40, inertia=.5, params = (2, 2, 0), dimensions = (100, 100), maxvelocity=1, Q = Problem1, inertiaDecreaseRate = 1):
		self._inertia = inertia
		self.params = params
		self.globalBestFitness = 0
		self.globalBest
		self.Q = Q
		self.inertiaDecreaseRate = inertiaDecreaseRate
		self.world = np.zeros(dimensions)
		self.swarm = createSwarm(npart, world, dimensions)
	def createSwarm(self, npart, world, dimensions):
		swarm = []
		for i in range(npart):
			pos = np.zeros(len(dimensions))
			for j in range(len(self.dimensions)):
				pos[j] = random.randint(0, dimensions[j])
			world[tuple(pos)) += 1
			particle = Particle(x, y, len(world[0]), len(world), Q)
			fitness = particle.getFitness()
			if fitness > self.globalBestFitness:
				self.globalBestFitness = fitness
				self.globalBestX = particle._x
				self.globalBestY = particle._y
			swarm.append(particle)
		return swarm
	# sets as neighbors the k closest:
	def setClosestNeighbors(self, k = 2):
		for i in swarm:
			#sort neighobrs into a stack
			sortedSwarm = sorted(swarm, key = (lambda other: i.Distance(other.pos), reversed = True)
			#the closest k particles that are not the target are its neighbors
			while len(i.neighbors) < k:
				neighbor = sortedSwarm.pop()
				if i != neighbor:
					i.neigbors.add(neighbor)
	def setEuclidianNeigbors(self, radius):		
		for i in self.swarm:
			i.neighbors = filter(lambda other: if i != other and i.distance(other.pos) < radius, self.swarm)
	def Update(self):
		#update everybody's position
		for i in self.swarm:
			world[tuple(i.pos)]-=1
			i.setVelocity(self._inertia, self.globalBest, self.params)
			i.scaleVelocity(self.maxvelocity)
			i.Position()
			world[tuple(i.pos)] += 1
		#Update everybody's fitness
		for i in self.swarm:
			i.updateFitness()
			if i.bestFitness > self.globalBestFitness:
				self.globalBestFitness = i.bestFitness
				self.globalBestX = i._bestX
				self.globalBestY = i._bestY
		self._inertia *= self.inertiaDecreaseRate