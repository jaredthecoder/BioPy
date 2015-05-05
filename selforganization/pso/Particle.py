import numpy as np
import random
from Problems import *

class Particle(object):
	#initializes the paritcle
	def __init__(self, pos = [0, 0], maxPos = [100, 100], Q = Problem1):
		self.pos = pos
		self.best = pos
		self.velocity = np.zeros(len(pos))
		self.maxPos = maxPos
		self.Q = Q
		self.neighbors = []
		self.bestFitness = 0
		self.curFitness = 0
	#sets the velocity of the particle accoriing to its topology
	def setVelocity(self, inertia, globalBest, params):
		print "setting Velocity"
		#get best neighbor
		best_neighbor = self
		for i in self.neighbors:
			if best_neighbor.curFitness < i.curFitness:
				best_neighbor = i
		for i in range(len(self.pos)):
			self.velocity[i] = inertia * self.velocity[i]
			self.velocity[i] += params[0] * random.random() * (self.best[i] - self.pos[i]) 
			self.velocity[i] += params[1] * random.random() * (globalBest[i]-self.pos[i]) 
			self.velocity[i] += params[2] * random.random() * (best_neighbor.pos[i] - self.pos[i])
			print self.velocity[i]
	#scales the velocity to the global maxixmum velocity
	def scaleVelocity(self, maxvelocity):
		print "scalling Velocity"
		change_distance = 0
		for i in self.velocity: #compute sum of velocities in all dimensions velocities
			change_distance+=i**2
		change_distance = change_distance**(1.0/2) *1.0
		for i in range(len(self.pos)): #find velocity for ith dimension
			if abs(self.velocity[i]) > maxvelocity**2:
				self.velocity[i] *= (maxvelocity/change_distance)
				print self.velocity[i]
	#adjusts the position of the swarm based on its current velocity	
	def Position(self):
		print "changing position"
		for i in range(len(self.pos)):
			self.pos[i] = (self.pos[i] + self.velocity[i])
	#evaluate and update fitness of particle
	def updateFitness(self):
		print "updating Fitness"
		self.curFitness = self.Q(self.pos, self.maxPos)
		if self.curFitness > self.bestFitness:
			self.bestFitness = self.curFitness
	#get distance between particle and position s
	def getDistance(self, s):
		distance = 0
		for i in range(len(s)):
			distance += ((self.pos[i]-s[i])%(maxes[i]/2))**2
		return distance ** (1.0/2)