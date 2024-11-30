import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

def type1():
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

  build_up = float(input("Enter build-up rate (degrees per 100m): "))

  d = []
  h = []

  r = 18000/(3.14*build_up)
  a = 0
  for [t_cor, s_cor] in zip(tar_co, sur_co):
      a += (t_cor - s_cor) ** 2
  a = np.sqrt(a)
  ht = a

  # Measured depth
  x = math.atan((ht - r) / (vt - kop))
  y = math.asin(r / ((1 / math.cos(x)) * (vt - kop)))
  alpha = x + y

  vc = r * math.sin(alpha)
  azi = math.atan((tar_co[0] - sur_co[0]) / (tar_co[1] - sur_co[1]))

  e = []
  n = []
  d = []
  e1 = []
  n1 = []
  d1 = []

  # For vertical section
  eitr = sur_co[1]
  nitr = sur_co[0]
  ditr = 0
  for i in range(kop + 1):
      e.append(eitr)
      n.append(nitr)
      d.append(ditr)
      ditr += 1

  # For build section
  vc = math.ceil(vc)
  for i in range(1, vc + 1):
      beta = math.asin(i / r)
      ditr += 1
      d.append(ditr)
      hitr = r - i * (1 / math.tan(beta))
      h.append(hitr)
      eitr = hitr * math.cos(azi)
      nitr = hitr * math.sin(azi)
      eitr = eitr + sur_co[1]
      nitr = nitr + sur_co[0]
      e.append(eitr)
      n.append(nitr)
      e1.append(eitr)
      n1.append(nitr)
      d1.append(ditr)

  # Hold profile
  vd = 0
  hx = 0
  for k in range(vt - kop - vc):
      vd = vt - kop - vc - k
      hx = ht - vd / (math.tan(1.570796 - alpha))
      ditr += 1
      d.append(ditr)
      h.append(hx)
      eitr = hx * math.cos(azi)
      nitr = hx * math.sin(azi)
      eitr = eitr + sur_co[1]
      nitr = nitr + sur_co[0]
      e.append(eitr)
      n.append(nitr)
  return e,n,d,e1,n1,d1


# plt.figure()  # Create a new figure
# ax = plt.axes(projection='3d')
# ax.plot(e, n, d)
# ax.plot(e1, n1, d1, color='k')
# plt.xlabel("East")
# plt.ylabel("North")
# ax.set_zlabel('Depth')
# ay = plt.gca()
# ay.invert_zaxis()
# plt.savefig("plot1.png")
# plt.close()