
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# vt = 6000
# ht = 5000
# build_up = 2
# sur_co = np.array([10, 5])
# tar_co = np.array([4331.51,2519.86])
# azi = 1.0472
def type4():
  vt = int(input("Enter vertical depth (vt): "))
  build_up = float(input("Enter build-up rate (degrees per 30m): "))

  # Surface coordinates input (two float values)
  surface_y = float(input("Enter surface East-coordinate: "))
  surface_x = float(input("Enter surface North-coordinate: "))
  sur_co = np.array([surface_x, surface_y])

  # Target coordinates input (two float values)
  target_y = float(input("Enter target East-coordinate: "))
  target_x = float(input("Enter target North-coordinate: "))
  tar_co = np.array([target_x, target_y])

  a = 0
  for [t_cor,s_cor] in zip(tar_co,sur_co):
    a += (t_cor-s_cor)**2
  a = np.sqrt(a)
  ht = a

  if tar_co[0] == sur_co[0]:
      slope = math.pi / 2  # 90 degrees in radians
  else:
      slope = math.atan((tar_co[1] - sur_co[1]) / (tar_co[0] - sur_co[0]))

  azi = 90 - slope
  azi = math.radians(azi)


  r = 18000/(3.14*build_up)

  r = math.ceil(r)
  kop = vt - r
  hx = ht - r
  hx = math.ceil(hx)
  e = []
  n = []
  d = []
  e1 = []
  n1 = []
  d1 = []
  h = []


  #Vertical drilling
  ditr = 0
  eitr = sur_co[1]
  nitr = sur_co[0]
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
    eitr = hitr*math.cos(azi)
    nitr = hitr*math.sin(azi)
    eitr = eitr + sur_co[1]
    nitr = nitr + sur_co[0]
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
    eitr = hitr*math.cos(azi)
    nitr = hitr*math.sin(azi)
    eitr = eitr + sur_co[1]
    nitr = nitr + sur_co[0]
    e.append(eitr)
    n.append(nitr)
  return e,n,d,e1,n1,d1


# ax = plt.axes(projection = '3d')
# ax.plot(e,n,d)
# ax.plot(e1,n1,d1,color = 'k')
# plt.xlabel("East")
# plt.ylabel("North")
# ax.set_zlabel('Depth')
# ay = plt.gca()
# ay.invert_zaxis()
# plt.show()
