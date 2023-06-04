Self-Organization
====

- [Particle Swarm Optimization](https://ieeexplore.ieee.org/document/488968)  
     Authors: James Kennedy and Russell Eberhart

Particle Swarm Optimization (PSO) is a population-based optimization algorithm inspired by the social behavior of bird flocking or fish schooling. It was originally proposed by Kennedy and Eberhart in 1995. The algorithm simulates the movement and interaction of a group of particles in a search space to find the optimal solution to a given optimization problem.

In PSO, each particle represents a potential solution to the problem and moves through the search space by adjusting its position and velocity. The particles "swarm" towards better regions of the search space based on their own experience and the collective knowledge of the swarm. The algorithm iteratively updates the particles' positions and velocities using two main components: personal best (pbest) and global best (gbest).

The personal best represents the best position that a particle has found so far in its search history. It is the position where the particle achieved its best objective function value. The global best represents the best position found by any particle in the entire swarm. It represents the overall best solution discovered by the swarm.

During each iteration of the algorithm, particles update their velocities and positions based on their current positions, velocities, pbest, and gbest. The update formula takes into account the particle's previous velocity, its attraction towards its pbest, and its attraction towards the gbest. By adjusting their velocities and positions, particles explore the search space and converge towards promising regions that are likely to contain the optimal solution.

PSO is known for its simplicity and ease of implementation. It has been successfully applied to a wide range of optimization problems, including continuous, discrete, and combinatorial problems. It is particularly effective in solving problems with non-linear and non-convex objective functions, where traditional optimization techniques may struggle.

However, PSO also has some limitations. It can suffer from premature convergence, where the swarm gets trapped in a suboptimal solution and fails to explore other promising regions. Various modifications and variants of PSO have been proposed to mitigate this issue, such as using adaptive parameters, introducing diversity maintenance mechanisms, or incorporating problem-specific knowledge.

Overall, PSO is a powerful and versatile optimization algorithm that leverages the collective intelligence of a swarm to efficiently explore and exploit the search space. It has found applications in various domains, including engineering, finance, data mining, and machine learning.