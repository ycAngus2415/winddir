import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

# we want to plot the arrows represent wind direction
def plot_quiver(imag, x, y, u, v):
    plt.imshow(imag, 'Greys')
    plt.quiver(x,y,u,v)
    plt.show()

if __name__ == '__main__':
   x=np.random.rand(5, 5)
   fig, az = plt.subplots()
   patch = []
   patch.append(Circle((3, 4), 2))
   collec = PatchCollection(patch)
   az.add_collection(collec)
   az.quiver(3, 4, 2, 5)
   plt.show()
