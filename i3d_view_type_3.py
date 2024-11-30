import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# ht = 1500
# vt = 10000
# kop = 7000
# sur_co = np.array([25,50])
# tar_co = np.array([156.55, 1544.22])
# azi = 1.0472
def type3():
  vt = int(input("Enter vertical depth (vt): "))
  kop = int(input("Enter kickoff point (kop): "))

  # Surface coordinates input (two float values)
  surface_x = float(input("Enter surface North-coordinate: "))
  surface_y = float(input("Enter surface East-coordinate: "))
  sur_co = np.array([surface_x, surface_y])

  # Target coordinates input (two float values)
  target_x = float(input("Enter target North-coordinate: "))
  target_y = float(input("Enter target East-coordinate: "))
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
  eitr = sur_co[1]
  nitr = sur_co[0]

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
    eitr = hitr*math.cos(azi)
    nitr = hitr*math.sin(azi)
    eitr = eitr + sur_co[1]
    nitr = nitr + sur_co[0]
    e.append(eitr)
    n.append(nitr)
    
    e1.append(eitr)
    n1.append(nitr)
    d1.append(ditr)
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


