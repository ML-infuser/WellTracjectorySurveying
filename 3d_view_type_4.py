
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# vt = 6000
# ht = 5000
# build_up = 2
# sur_co = np.array([15.32, 5.06])
# azi = 1.0472
# tar_co = np.array([sur_co[0] + ht*math.sin(azi),sur_co[1] + ht*math.cos(azi)])


vt = int(input("Enter vertical depth (vt): "))
ht = int(input("Enter horizontal displacement (ht): "))
build_up = float(input("Enter build-up rate (degrees per 30m): "))

# # Surface coordinates input (two float values)
surface_x = float(input("Enter surface East-coordinate: "))
surface_y = float(input("Enter surface North-coordinate: "))
sur_co = np.array([surface_x, surface_y])
azi = float(input("Enter azimuth in degrees "))
azi = math.radians(azi)


r = 18000/(3.14*build_up)

r = math.ceil(r)
kop = vt - r
hx = ht - r

e = []
n = []
d = []
e1 = []
n1 = []
d1 = []
h = []


#Vertical drilling
ditr = 0
eitr = sur_co[0]
nitr = sur_co[1]
for i in range(kop):
    d.append(ditr)
    e.append(eitr)
    n.append(nitr)
    ditr += 1


#Build Section

for i in range(r):
  d.append(ditr)
  ditr += 1
  x = math.asin(i/r)
  hitr = r - r*math.cos(x)
  h.append(hitr)
  eitr = hitr*math.sin(azi)
  nitr = hitr*math.cos(azi)
  eitr = eitr + sur_co[0]
  nitr = nitr + sur_co[1]
  e.append(eitr)
  n.append(nitr)
  e1.append(eitr)
  n1.append(nitr)
  d1.append(ditr)
  

# Horizontal Drilling
for k in range(hx):
  d.append(ditr)
  hitr += 1
  h.append(hitr)
  eitr = hitr*math.sin(azi)
  nitr = hitr*math.cos(azi)
  eitr = eitr + sur_co[0]
  nitr = nitr + sur_co[1]
  e.append(eitr)
  n.append(nitr)


ax = plt.axes(projection = '3d')
ax.plot(e,n,d)
ax.plot(e1,n1,d1,color = 'k')
plt.xlabel("East")
plt.ylabel("North")
ax.set_zlabel('Depth')
ay = plt.gca()
ay.invert_zaxis()

df = pd.DataFrame({'e': e,'n': n,  'd':d})
csv_file_path = 'type_4.csv'
df.to_csv(csv_file_path, index=False)  

# plt.show()
