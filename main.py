import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
import math
from mpl_toolkits.mplot3d import Axes3D

def calculate_distance(point1, point2):
    """Calculate the 3D distance between two points"""
    return np.sqrt(np.sum((point1 - point2) ** 2))

def are_wells_shared_surface(well1_start, well2_start, tolerance=1.0):
    """
    Check if two wells start from the same surface location
    tolerance: maximum distance in feet to consider points as same location
    """
    return calculate_distance(well1_start, well2_start) < tolerance

def find_intersection_zones(points1, points2, kop1, kop2, shared_surface, threshold=20):
    """
    Find zones of intersection between two well trajectories, 
    considering kickoff points for shared surface wells.
    """
    zones = []
    current_zone = None
    tree2 = KDTree(points2)
    
    # Determine start index for collision checking
    start_idx = 0 if not shared_surface else max(kop1, kop2)
    
    for i in range(start_idx, len(points1)):
        point = points1[i]
        distances, indices = tree2.query(point, k=1)
        
        # Only consider points after kickoff for the second well if shared surface
        if shared_surface and indices < kop2:
            continue
            
        if distances < threshold:
            close_point = points2[indices]
            
            if current_zone is None:
                # Start new zone
                current_zone = {
                    'well1_start': point,
                    'well2_start': close_point,
                    'well1_start_idx': i,
                    'well2_start_idx': indices,
                    'well1_end': point,
                    'well2_end': close_point,
                    'well1_end_idx': i,
                    'well2_end_idx': indices
                }
            else:
                # Update end of current zone
                current_zone['well1_end'] = point
                current_zone['well2_end'] = close_point
                current_zone['well1_end_idx'] = i
                current_zone['well2_end_idx'] = indices
        elif current_zone is not None:
            # End of zone reached
            zones.append(current_zone)
            current_zone = None
    
    # Don't forget to add the last zone if it exists
    if current_zone is not None:
        zones.append(current_zone)
    
    return zones

# Get input from user
n = int(input("How many types you want to compare: "))
types = []

print("Enter types (1-5): ")
for i in range(n):
    val = int(input())
    if val < 1 or val > 5:
        print("Invalid Type")
        exit()
    types.append(val)

# Run scripts for each type
for val in types:
    subprocess.run(["python", f"3d_view_type_{val}.py"])

# Set up 3D plot
plt.figure(figsize=(12, 8))
ax = plt.axes(projection='3d')
ax.set_xlabel("East")
ax.set_ylabel("North")
ax.set_zlabel('Depth')
ax.invert_zaxis()

colors = ['blue', 'green', 'red', 'purple', 'orange']
markers = ['o', 's', '^', 'D', 'v']

# Load data and store kickoff points
data = {}
kickoff_points = {}
for val in types:
    try:
        df = pd.read_csv(f'type_{val}.csv')
        data[val] = df
        
        # Find kickoff point (first point where trajectory deviates from vertical)
        points = np.column_stack((df['e'], df['n'], df['d']))
        start_point = points[0]
        
        # Find first point that deviates from vertical by checking horizontal distance
        for i, point in enumerate(points):
            if calculate_distance(point[:2], start_point[:2]) > 1.0:  # 1 foot tolerance
                kickoff_points[val] = i
                break
        else:
            kickoff_points[val] = 0
        
        # Plot the well trajectory
        ax.plot(df['e'], df['n'], df['d'], 
                color=colors[val-1], 
                label=f'Well Type {val}',
                linewidth=2)
        
        # Plot start, kickoff, and end points
        ax.scatter(df['e'].iloc[0], df['n'].iloc[0], df['d'].iloc[0], 
                  color=colors[val-1], marker=markers[val-1], s=100, label=f'Start Well {val}')
        ax.scatter(df['e'].iloc[kickoff_points[val]], df['n'].iloc[kickoff_points[val]], 
                  df['d'].iloc[kickoff_points[val]], 
                  color='black', marker='x', s=100, label=f'KOP Well {val}')
        ax.scatter(df['e'].iloc[-1], df['n'].iloc[-1], df['d'].iloc[-1], 
                  color=colors[val-1], marker=markers[val-1], s=100)
        
    except FileNotFoundError:
        print(f"Warning: type_{val}.csv not found")
        continue

# Check for intersections between each pair of wells
collision_threshold = 20  # feet
intersection_zones_found = False

for i, val1 in enumerate(types):
    for val2 in types[i+1:]:  # Only compare each pair once
        points1 = np.column_stack((data[val1]['e'], data[val1]['n'], data[val1]['d']))
        points2 = np.column_stack((data[val2]['e'], data[val2]['n'], data[val2]['d']))
        
        # Check if wells share surface location
        shared_surface = are_wells_shared_surface(points1[0], points2[0])
        
        print(f"\nAnalyzing intersection between Well {val1} and Well {val2}:")
        print(f"Wells {'share' if shared_surface else 'do not share'} surface location")
        print(f"Kickoff points - Well {val1}: {kickoff_points[val1]}, Well {val2}: {kickoff_points[val2]}")
        
        zones = find_intersection_zones(points1, points2, 
                                     kickoff_points[val1], 
                                     kickoff_points[val2], 
                                     shared_surface,
                                     collision_threshold)
        
        if zones:
            intersection_zones_found = True
            print(f"Found {len(zones)} intersection zone(s):")
            
            for idx, zone in enumerate(zones, 1):
                # Calculate zone length
                zone_length = calculate_distance(zone['well1_start'], zone['well1_end'])
                
                print(f"\nIntersection Zone {idx}:")
                print(f"Start point - Well {val1}: E={zone['well1_start'][0]:.2f}, "
                      f"N={zone['well1_start'][1]:.2f}, D={zone['well1_start'][2]:.2f}")
                print(f"Start point - Well {val2}: E={zone['well2_start'][0]:.2f}, "
                      f"N={zone['well2_start'][1]:.2f}, D={zone['well2_start'][2]:.2f}")
                print(f"End point - Well {val1}: E={zone['well1_end'][0]:.2f}, "
                      f"N={zone['well1_end'][1]:.2f}, D={zone['well1_end'][2]:.2f}")
                print(f"End point - Well {val2}: E={zone['well2_end'][0]:.2f}, "
                      f"N={zone['well2_end'][1]:.2f}, D={zone['well2_end'][2]:.2f}")
                print(f"Zone length: {zone_length:.2f} feet")
                
                # Plot intersection zone markers
                ax.scatter(zone['well1_start'][0], zone['well1_start'][1], zone['well1_start'][2],
                          color='red', s=100, edgecolor='black', 
                          marker='o', label='Intersection Start' if idx==1 else "")
                ax.scatter(zone['well1_end'][0], zone['well1_end'][1], zone['well1_end'][2],
                          color='yellow', s=100, edgecolor='black', 
                          marker='s', label='Intersection End' if idx==1 else "")

if not intersection_zones_found:
    print("\nNo intersection zones detected between wells.")

# Enhance the plot
ax.legend()
ax.grid(True)
plt.title("Well Trajectory Analysis")

# Add text box with analysis parameters
info_text = (f"Analysis Parameters:\n"
             f"Collision Threshold: {collision_threshold} ft\n"
             f"Surface Sharing Tolerance: 1.0 ft")
plt.figtext(0.02, 0.02, info_text, fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

# Adjust the view
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()