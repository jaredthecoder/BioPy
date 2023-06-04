import math

def mdist(maxX, maxY):
    '''
    Calculate the Manhattan distance from the origin to the maximum coordinates.

    Parameters:
        maxX (float): Maximum X-coordinate.
        maxY (float): Maximum Y-coordinate.

    Returns:
        float: Manhattan distance.

    Example:
        mdist(5, 8)  # Returns: 9.899494936611665
    '''
    return math.sqrt(maxX ** 2 + maxY ** 2 / 2)

def pdist(px, py):
    '''
    Calculate the Euclidean distance between the given point and (20, 7).

    Parameters:
        px (float): X-coordinate of the point.
        py (float): Y-coordinate of the point.

    Returns:
        float: Euclidean distance.

    Example:
        pdist(10, 3)  # Returns: 12.083045973594572
    '''
    return math.sqrt((px - 20) ** 2 + (py - 7) ** 2)

def ndist(px, py):
    '''
    Calculate the Euclidean distance between the given point and (-20, -7).

    Parameters:
        px (float): X-coordinate of the point.
        py (float): Y-coordinate of the point.

    Returns:
        float: Euclidean distance.

    Example:
        ndist(-10, -3)  # Returns: 12.083045973594572
    '''
    return math.sqrt((px + 20) ** 2 + (py + 7) ** 2)

def Problem1(pos, maxes):
    '''
    Solve Problem 1 based on the given position and maximum coordinates.

    Parameters:
        pos (list): List containing X and Y coordinates of the position.
        maxes (list): List containing maximum X and Y coordinates.

    Returns:
        float: Solution to Problem 1.

    Example:
        Problem1([10, 5], [5, 8])  # Returns: 100.0
    '''
    return 100 * (1 - pdist(pos[0], pos[1]) / mdist(maxes[0], maxes[1]))

def Problem2(pos, maxes):
    '''
    Solve Problem 2 based on the given position and maximum coordinates.

    Parameters:
        pos (list): List containing X and Y coordinates of the position.
        maxes (list): List containing maximum X and Y coordinates.

    Returns:
        float: Solution to Problem 2.

    Example:
        Problem2([10, 5], [5, 8])  # Returns: 89.89827550319647
    '''
    pd = pdist(pos[0], pos[1])
    nd = ndist(pos[0], pos[1])
    md = mdist(maxes[0], maxes[1])

    ret = 9 * max(0, 10 - pd ** 2)
    ret += 10 * (1 - pd / md)
    ret += 70 * (1 - nd / md)
    return ret
