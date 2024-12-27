import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

# vt = 9880
# kop = 4650
# sur_co = np.array([0, 0])
# ht = 5500
# azi = 0.349066
# tar_co = np.array([sur_co[0] + ht*math.sin(azi),sur_co[1] + ht*math.cos(azi)])
# build_up = 1.5


vt = int(input("Enter vertical depth (vt): "))
kop = int(input("Enter kickoff point (kop): "))

# Surface coordinates input (two float values)

surface_x = float(input("Enter surface East-coordinate: "))
surface_y = float(input("Enter surface North-coordinate: "))
sur_co = np.array([surface_x, surface_y])

ht = float(input("Enter horizontal distance: "))
build_up = float(input("Enter build-up rate (degrees per 100m): "))
azi = float(input("Enter azimuth in degrees "))
azi = math.radians(azi)


d = []
h = []

r = 18000/(3.14*build_up)

# Measured depth
x = 0
y = 0
x = math.atan((ht-r)/(vt - kop))
y = math.asin(r/((1/math.cos(x)) * (vt-kop)))
alpha = x + y

vc = r*math.sin(alpha)

e = []
n = []
d = []

e1 = []
n1 = []
d1 = []


#for vertical section
eitr = sur_co[0]
nitr = sur_co[1]
ditr = 0
for i in range(kop +1):
    e.append(eitr)
    n.append(nitr)
    d.append(ditr)
    ditr += 1
 
# for build section
vc = math.ceil(vc)

for i in range(1,vc+1):
  beta = math.asin(i / r)

  ditr += 1
  d.append(ditr)
  hitr = r - i*(1/math.tan(beta))
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
   

# hold profile
vd = 0
hx = 0
for k in range(vt-kop-vc):
  vd = vt - kop - vc -k
  hx = ht - vd /(math.tan(1.570796 - alpha))
  ditr += 1
  d.append(ditr)
  h.append(hx)
  eitr = hx*math.sin(azi)
  nitr = hx*math.cos(azi)
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
csv_file_path = 'type_1.csv'
df.to_csv(csv_file_path, index=False)  
# plt.show()
