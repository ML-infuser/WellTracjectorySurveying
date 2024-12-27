import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# ht = 1500
# vt = 10000
# kop = 7000
# sur_co = np.array([5,15])
# azi = 1.0472
# tar_co = np.array([sur_co[0] + ht*math.sin(azi),sur_co[1] + ht*math.cos(azi)])

vt = int(input("Enter vertical depth (vt): "))
kop = int(input("Enter kickoff point (kop): "))

# # Surface coordinates input (two float values)
surface_x = float(input("Enter surface East-coordinate: "))
surface_y = float(input("Enter surface North-coordinate: "))
sur_co = np.array([surface_x, surface_y])
ht = int(input("Enter horizontal depth (ht): "))
azi = float(input("Enter azimuth in degrees "))
azi = math.radians(azi)

h = []
e = []
n = []
d = []
e1 = []
n1 = []
d1 = []

p = vt - kop
alpha = 2*math.atan(ht/(vt-kop))
r = (vt - kop)/math.sin(alpha)

ditr = 0
hitr = 0
eitr = sur_co[0]
nitr = sur_co[1]

#Vertical drilling
for i in range(kop):
    e.append(eitr)
    n.append(nitr)
    d.append(ditr)
    ditr += 1

#Build section
for i in range(p):
  x = math.asin(i/r)
  hitr = r - r*math.cos(x)
  h.append(hitr)
  d.append(ditr)
  ditr += 1
  eitr = hitr*math.sin(azi)
  nitr = hitr*math.cos(azi)
  eitr = eitr + sur_co[0]
  nitr = nitr + sur_co[1]
  e.append(eitr)
  n.append(nitr)
  
  e1.append(eitr)
  n1.append(nitr)
  d1.append(ditr)

ax = plt.axes(projection = '3d')
ax.plot(e,n,d)
ax.plot(e1,n1,d1,color = 'k')
plt.xlabel("East")
plt.ylabel("North")
ax.set_zlabel('Depth')
ay = plt.gca()
ay.invert_zaxis()

df = pd.DataFrame({'e': e,'n': n,  'd':d})
csv_file_path = 'type_3.csv'
df.to_csv(csv_file_path, index=False)  

# plt.show()


