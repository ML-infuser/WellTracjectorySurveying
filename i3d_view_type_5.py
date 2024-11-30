import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# vt = 9000
# ht = 12000
# kop = 2000
# build_up = 2
# final_incl = 0.523599
# initial_incl = 1.0472
# azi = 1.0472
# sur_co = np.array([15, 5])
# tar_co = np.array([6000,10406.91])
def type5():
  vt = int(input("Enter vertical depth (vt): "))
  kop = int(input("Enter kickoff point (kop): "))
  build_up = float(input("Enter build-up rate (degrees per 30m): "))
  final_incl = float(input("Enter final inclination (in radians): "))
  initial_incl = float(input("Enter initial inclination (in radians): "))


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

  e = []
  n = []
  d = []
  e1 = []
  n1 = []
  d1 = []
  e2 = []
  n2 = []
  d2 = []
  h = []

  r1 = 18000/(3.14*build_up)
  r2 = r1
  vc = r1*math.sin(initial_incl)
  ve = r2 - r2*math.cos(final_incl)
  vd = vt - kop - vc - ve
  vc = math.ceil(vc)
  ve = math.ceil(ve)
  vd = math.ceil(vd)

  ditr = 0
  hitr = 0

  # Vertical Drilling
  eitr = sur_co[1]
  nitr = sur_co[0]
  for i in range(kop):
    d.append(ditr)
    e.append(eitr)
    n.append(nitr)
    ditr += 1


  # First Build Section
  for i in range(vc):
    d.append(ditr)
    ditr += 1
    x = math.asin(i/r1)
    hitr = r1 - r1*math.cos(x)
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
    

  # Hold section
  hc = h[-1]
  for i in range(vd):
    d.append(ditr)
    ditr += 1
    hitr = hc + i *math.tan(initial_incl)
    h.append(hitr)
    eitr = hitr*math.cos(azi)
    nitr = hitr*math.sin(azi)
    eitr = eitr + sur_co[1]
    nitr = nitr + sur_co[0]
    e.append(eitr)
    n.append(nitr)
    

  # Build Section
  hd = h[-1]
  for i in range(ve):
    d.append(ditr)
    ditr += 1
    de = r2*math.cos(final_incl) + i
    he = r2*math.sin(final_incl) - (r2**2 - de**2)**0.5
    hitr = hd + he
    h.append(hitr)
    eitr = hitr*math.cos(azi)
    nitr = hitr*math.sin(azi)
    eitr = eitr + sur_co[1]
    nitr = nitr + sur_co[0]
    e.append(eitr)
    n.append(nitr)
    e2.append(eitr)
    n2.append(nitr)
    d2.append(ditr)
    

  # Horizontal Section
  he = h[-1]
  hx = ht - he
  hx = math.ceil(hx)
  for i in range(hx):
    d.append(ditr)
    hitr += 1
    h.append(hitr)
    
    eitr = hitr*math.cos(azi)
    nitr = hitr*math.sin(azi)
    eitr = eitr + sur_co[1]
    nitr = nitr + sur_co[0]
    e.append(eitr)
    n.append(nitr)
  return e,n,d,e1,n1,d1,e2,n2,d2

# ax = plt.axes(projection = '3d')
# ax.plot(e,n,d)
# ax.plot(e1,n1,d1,color = 'k')
# ax.plot(e2,n2,d2,color = 'k')
# plt.xlabel("East")
# plt.ylabel("North")
# ax.set_zlabel('Depth')
# ay = plt.gca()
# ay.invert_zaxis()
# plt.show()