import numpy as np
import random
from Problems import *
from Particle import *
class PSO(object):
	'''Particle Swarm Optimization:
		Fields:
			inertia: the inertia of the world
			inertiaPrime: rate of change of a the inertia (a percentage)
			npart: number of particles in the world
			phi: The parameters of the world. Conscious, social and a potential 3rd neigborhood parameter
			dimensions: an array of n elements each representing the range for a dimension in the world
			maxvelocity: the maximum possible velocity of a particle in the world
			Q: Fitness function of the swarm
			swarm: the swarm of particles
		Fuctions:
			Update: moves all particles swarms to the next epoch
			setClosestNeighbors: for each particle sets as neighbors the k closest particles to it
			setEuclidianNeigbhbors: for each particle sets its neighbors as all particles within range
			getError: returns the error for the given orientation.
	'''
	def __init__(self, npart = 40, inertia=.5, phi = (2, 2, 2), dimensions = (100, 100), maxvelocity=1, Q = Problem1, inertiaPrime = 1):
		self._inertia = inertia
		self.phi = phi
		self.globalBestFitness = 0
		self.maxvelocity = 1
		self.globalBest = None
		self.Q = Q
		self.inertiaPrime = inertiaPrime
		self.swarm = []
		#Create and initialize the swarm
		for i in range(npart):
			pos = np.zeros(len(dimensions))
			for j in range(len(dimensions)):
				pos[j] = random.randint(-1 * dimensions[j]/2, dimensions[j]/2)
			particle = Particle(pos, dimensions)
			fitness = particle.updateFitness(Q)
			if fitness > self.globalBestFitness or self.globalBestFitness == 0:
				self.globalBestFitness = fitness
				self.globalBest = particle.pos
			self.swarm.append(particle)
	def setClosestNeighbors(self, k):
		if k > 0:
			for i in self.swarm:
				#sort neighobrs into a stack
				sortedSwarm = sorted(self.swarm, key = (lambda other: i.getDistance(other.pos)), reverse = True)
				#the closest k particles that are not the target are its neighbors
				while len(i.neighborhood) < k:
					neighbor = sortedSwarm.pop()
					if i != neighbor:
						i.neighborhood.append(neighbor)
				print i.neighborhood
	#neighborhood is all particles within a certain range
	def setEuclidianNeigbors(self, radius):
		if radius > 0:
			for i in self.swarm:
				i.neighborhood = filter(lambda other: i != other and i.getDistance(other.pos) < radius, self.swarm)
				print i.neighborhood
	#updates the positions of all particles
	def Update(self):
		#update everybody's position
		for i in self.swarm:
			i.setVelocity(self._inertia, self.globalBest, self.phi)
			i.scaleVelocity(self.maxvelocity)
			i.Move()
		#Update everybody's fitness
		for i in self.swarm:
			i.updateFitness(self.Q)
			if i.bestFitness > self.globalBestFitness:
				self.globalBestFitness = i.bestFitness
				self.globalBest = i.best
		#degreade inertia
		self._inertia *= self.inertiaPrime
	#return the error of the swarm at the given orientation
	def getError(self):
		error = np.zeros(len(self.swarm[0].pos));
		for i in self.swarm:
			error += (i.pos - self.globalBest)**2
		error = (1.0/(2*len(self.swarm))*error)**(.5)
		return error
