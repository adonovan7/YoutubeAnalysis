import matplotlib.pyplot as plt
import os 
warnings.filterwarnings('ignore')

path = '/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/'

os.chdir('/Users/andiedonovan/myProjects/Youtube_Python_Project/AndiesBranch/scripts/') # change directory

import UserDataModel.py

# Data Visual 1: Stacked Plot
models = [1,2,3,4,5]

positive = [7,8,6,11,7]
negative =   [2,3,4,3,2]
neutral =  [7,8,7,2,2]


plt.plot([],[],color='m', label='positive', linewidth=5)
plt.plot([],[],color='c', label='negative', linewidth=5)
plt.plot([],[],color='k', label='neutral', linewidth=5)

plt.stackplot(models, negative, neutral, positive, colors=['m','c','k'])

plt.xlabel('Model')
plt.ylabel('Sentiment Distribution')
plt.title('Distribution of Sentiments Ratiosby Model')
plt.legend()
plt.show()



# Data Visual 2: Comparison of Your Video to our Classified Data

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

x = [1,2,3,4,5,6,7,8,9,10]
y = [5,6,7,8,2,5,6,3,7,2]
z = [1,2,6,3,2,7,3,3,7,2]

x2 = [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]
y2 = [-5,-6,-7,-8,-2,-5,-6,-3,-7,-2]
z2 = [1,2,6,3,2,7,3,3,7,2]

ax1.scatter(x, y, z, c='g', marker='o')
ax1.scatter(x2, y2, z2, c ='r', marker='o')

ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')

plt.show()