import numpy as np
import random
from Problems import *

class Particle(object):
	'''Particle
		A representation of a particle in the PSO.
		Fields:
			pos: The position of the particle in the world array, represented as an array.
				The particles actual xyz... cordinate is equal to: pos - maxPos/2
			best: The position visited by this particle that has the best fitness
			curFitness: the fitness of the particle at its current position
			bestFitness: the best fitness of any position the particle has traveled to.
				In other words, the fitness coresponding to the "best" position field.
			Q: The fitness equation of the particle. Takes the pos and the max position
			velocity: a vector that represents the current velocity and direction of the particle
			maxPos: array of he maximum possible position of the particle in any dimension i.
			neighbors: a list of other particles considered "neighbors" to the given particle

		Functions:
			setVelocity:
				parameters: inertia, globalBest, phi
				Description: determines the velocity of the particle
			scaleVelocity:
				parameters: maxvelocity
				Description: rescales the particle velocity so that it does not
					exceed the global maxvelocity or exit the world range
			Move:
				parameters: none
				Description: updates the particles position according to its curent velocity
			updateFitness:
				parameters: Fitness
				Description: caclulates the fitness of the particle at its current position
					updates bestFitness and best fields if necessary
			getDistance:
				parameters: position s
				Description: Finds the distance between particle and point at position s
	'''
	#initializes the paritcle
	def __init__(self, pos = [0, 0], maxPos = [100, 100]):
		self.pos = pos
		self.best = pos
		self.curFitness = 0
		self.bestFitness = 0
		self.velocity = np.zeros(len(pos))
		self.maxPos = np.array(maxPos)
		self.neighborhood = []
	#sets the velocity of the particle accoriing to its topology
	def setVelocity(self, inertia, globalBest, phi):
		#get best neighbor
		best_neighbor = self
		for i in self.neighborhood:
			if best_neighbor.curFitness < i.curFitness:
				best_neighbor = i
		for i in range(len(self.pos)):
			self.velocity[i] = inertia * self.velocity[i]
			self.velocity[i] += phi[0] * random.random() * (self.best[i] - self.pos[i])
			self.velocity[i] += phi[1] * random.random() * (globalBest[i]-self.pos[i])
			self.velocity[i] += phi[2] * random.random() * (best_neighbor.pos[i] - self.pos[i])
	#scales the velocity to the global maxixmum velocity
	def scaleVelocity(self, maxvelocity):
		change_distance = 0
		for i in self.velocity: #compute sum of velocities in all dimensions velocities
			change_distance+=i**2
		change_distance = change_distance**(1.0/2) *1.0
		for i in range(len(self.pos)): #find velocity for ith dimension
			if abs(self.velocity[i]) > maxvelocity**2:
				self.velocity[i] *= (maxvelocity/change_distance)
	#adjusts the position of the swarm based on its current velocity
	def Move(self):
		for i in range(len(self.pos)):
			self.pos[i] = (self.pos[i] + self.velocity[i])
	#evaluate and update fitness of particle
	def updateFitness(self, Q):
		#Find fitness at the current position, scaled to the range of the map
		self.curFitness = Q(self.pos, self.maxPos)
		if self.curFitness > self.bestFitness:
			self.bestFitness = self.curFitness
	#get distance between particle and position s (Wraps around)
	def getDistance(self, s):
		x = self.pos - s
		x = np.minimum(x, (self.maxPos)-x)
		return sum((x)**2)**.5
