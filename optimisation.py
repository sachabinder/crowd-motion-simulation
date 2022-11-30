import numpy as np
import cvxpy as cp
from utils import RectangleObstacle

def distance_between_people(positions: np.ndarray, people_radius: int) -> np.ndarray:
    distances = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distances.append(
                np.linalg.norm(positions[i] - positions[j]) - 2 * people_radius
            )
    return np.array(distances)

def distance_walls (positions: np.ndarray, people_radius: int, width: int, height: int) -> np.ndarray:
    distances = []
    for k in range(len(positions)):
        distances.append(positions[k][0] - people_radius,
                        positions[k][1] - people_radius,
                        width - positions[k][0] - people_radius,
                        height - positions[k][1] - people_radius)
    return np.array(distances)

def distance_obstacle (positions: np.ndarray, people_radius: int, obstacle: RectangleObstacle) -> np.ndarray:
    
    """We have to check in which zone the people are to compute their distance to the obstacle
        Zone 2  |       Zone 3          | Zone 4
                |                       |
        --------+-----------------------|--------
        Zone 1  |        Obstacle       | Zone 5
                |       (Rectangle)     |
        --------+-----------------------+--------
        Zone 8  |       Zone 7          | Zone 6
                |                       |
    """
    
    distances = []
    for k in range(len(positions)):

        x = positions[k][0]
        y = positions[k][1]      

        #Check Zone 1 and 5
        if (obstacle.x_min <= x <= obstacle.x_max) :
            if (y<= obstacle.y_min): #Zone 1
                distances.append(abs(y - obstacle.y_min))
            elif (y>= obstacle.y_max): #Zone 5
                distances.append(abs(y - obstacle.y_max))
            else:
                return "error: a person is in an obstacle..."
        
        #Check Zone 1 and 5
        elif (obstacle.y_min <= y <= obstacle.y_max) :
            if (x<= obstacle.x_min): #Zone 3
                distances.append(abs(x - obstacle.x_min))
            elif (x>= obstacle.x_max): #Zone 7
                distances.append(abs(x - obstacle.x_max))
            else:
                return "error: a person is in an obstacle..."
        
        #Check Zone 2 and 4
        elif (x < obstacle.x_min) :
            if (y< obstacle.y_min): #Zone 2
                distances.append(
                    np.linalg.norm(positions[k] - np.array([obstacle.x_min, obstacle.y_min]))
                    )
            elif (y> obstacle.y_max): #Zone 4
                distances.append(
                    np.linalg.norm(positions[k] - np.array([obstacle.x_max, obstacle.y_min]))
                    )

        #Check Zone 6 and 8
        elif (x > obstacle.x_max) :
            if (y< obstacle.y_min): #Zone 8
                distances.append(
                    np.linalg.norm(positions[k] - np.array([obstacle.x_max, obstacle.y_min]))
                    )
            elif (y> obstacle.y_max): #Zone 6
                distances.append(
                    np.linalg.norm(positions[k] - np.array([obstacle.x_max, obstacle.y_max]))
                    )
    
    return distances
    

