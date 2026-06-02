import numpy as np

def capillary_vector(wall, theta):
    """
    Returns capillary vector m at wall.
    
    Bottom/Top: perpendicular to wall
    Left/Right: angled by contact angle theta into fluid
    """
    if wall == 'bottom':
        return np.array([0, 1])  # perpendicular
    elif wall == 'top':
        return np.array([0, -1])  # perpendicular
    elif wall == 'left':
        n = np.array([1, 0])
        t = np.array([0, 1])
        m = n * np.cos(theta) - t * np.sin(theta)
        m /= np.linalg.norm(m)
        return m
    elif wall == 'right':
        n = np.array([-1, 0])
        t = np.array([0, 1])
        m = n * np.cos(theta) - t * np.sin(theta)
        m /= np.linalg.norm(m)
        return m
    else:
        raise ValueError("Wall must be 'bottom','top','left','right'")

# Example usage
theta = np.pi/4  # 45°
walls = ['bottom', 'top', 'left', 'right']
for wall in walls:
    print(f"{wall.capitalize()} wall vector:", capillary_vector(wall, theta))