import math
def mdist(maxX, maxY):
	return math.sqrt(maxX**2 + maxY**2/2)
def pdist(px, py):
	return math.sqrt((px-20)**2 + (py-7)**2) 
def ndist(px, py):
	return math.sqrt((px+20)**2 + (py+7)**2) 
def Problem1(pos, maxes):
	return 100*(1-pdist(pos[0], pos[1])/mdist(maxes[0], maxes[1]))
def Problem2(pos, maxes):
	pd = pdist(pos[0], pos[1])
	nd = ndist(pos[0], pos[1])
	md = mdist(maxes[0], maxes[1])

	ret = 9*max(0, 10-pd**2)
	ret+=10*(1-pd/md)
	ret+=70*(1-nd/md)
	return ret