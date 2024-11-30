import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
import math
from i3d_view_type_1 import type1
from i3d_view_type_2 import type2
from i3d_view_type_3 import type3
from i3d_view_type_4 import type4
from i3d_view_type_5 import type5

def calculate_angle(v1, v2):
    """Calculate the angle between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)

def has_curve(points, threshold=10):
    """Detect if the curve starts based on angle changes."""
    for i in range(1, len(points)-1):
        v1 = points[i] - points[i-1]
        v2 = points[i+1] - points[i]
        angle = calculate_angle(v1, v2)
        if angle > threshold:
            return True, i
    return False, None

n = int(input("How many types you want to compare: "))
types = []

print("Enter types: ")
for i in range(n):
    val = int(input())
    if val < 1 or val > 5:
        print("Invalid Type")
        exit()
    types.append(val)

# Run scripts for each type
# for val in types:
#     subprocess.run(["python", f"i3d_view_type_{val}.py"])

# Set up 3D plot
colors = ['blue', 'green', 'red', 'purple', 'orange']

# Load data into dictionaries and KDTree for faster lookups
data = {}
trees = {}
all_lines = []  # Store the line data for combining later

for val in types:
    df = pd.read_csv(f'type_{val}.csv')
    data[val] = df
    points = np.column_stack((df['n'], df['e'], df['d']))
    trees[val] = KDTree(points)
    all_lines.append(points)  # Store the points for later use

# Create separate figures first
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("East")
ax.set_ylabel("North")
ax.set_zlabel('Depth')
ax.invert_zaxis()
figures_data = []
for idx, val in enumerate(types):
    if(val == 1):
        e2,n2,d2,e1,n1,d1 = type1()
        ax.plot(e2, n2, d2, color='k')
        ax.plot(e1, n1, d1, color='k')
    elif(val == 2):
        e3,n3,d3,e1,n1,d1,e2,n2,d2 = type2()
        ax.plot(e3,n3,d3,color = 'k')
        ax.plot(e1,n1,d1,color = 'k')
        ax.plot(e2,n2,d2,color = 'k')
    elif(val == 3):
        e2,n2,d2,e1,n1,d1 = type3()
        ax.plot(e2, n2, d2, color='k')
        ax.plot(e1, n1, d1, color='k')
    elif(val == 4):
        e2,n2,d2,e1,n1,d1 = type4()
        ax.plot(e2, n2, d2, color='k')
        ax.plot(e1, n1, d1, color='k')
    elif(val == 5):
        e2,n2,d2,e1,n1,d1,e3,n3,d3 = type4()
        ax.plot(e2, n2, d2, color='k')
        ax.plot(e1, n1, d1, color='k')
        ax.plot(e3, n3, d3, color='k')
    ax.set_title(f'Type {val}')
plt.show()
