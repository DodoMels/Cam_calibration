import numpy as np
import matplotlib.pyplot as plt
import json



def plot_radial_distortion(ax, k1, k2, k3):
	n = 20
	x = np.linspace(-100, 100, n)
	y = np.linspace(-100, 100, n)
	x, y = np.meshgrid(x, y)
	rsq = np.square(x) + np.square(y)
	k = k1*rsq + k2*rsq*rsq + k3*rsq*rsq*rsq
	dx = x*k
	dy = y*k
	ax.quiver(x, y, dx, dy)
	ax.set_aspect('equal')
	ax.set_title('Radial distortion')



def plot_tangential_distortion(ax, p1, p2):
	n = 20
	x = np.linspace(-100, 100, n)
	y = np.linspace(-100, 100, n)
	x, y = np.meshgrid(x, y)
	rsq = np.square(x) + np.square(y)
	xy = 2*x*y
	dx = p1*xy + p2*(rsq+2*np.square(x))
	dy = p2*xy + p1*(rsq+2*np.square(y))
	ax.quiver(x, y, dx, dy)
	ax.set_aspect('equal')
	ax.set_title('Tangential distortion')



with open('calib.json') as f:
	calib = json.load(f)
dist_coeffs = np.array(calib['dist_coeffs'][0])

fig = plt.figure()

ax = fig.add_subplot(121)
plot_radial_distortion(ax, dist_coeffs[0], dist_coeffs[1], dist_coeffs[4]) # From actual calibration
#plot_radial_distortion(ax, 0.002, 0.0, 0.0) # Artificial values

ax = fig.add_subplot(122)
plot_tangential_distortion(ax, dist_coeffs[2], dist_coeffs[3]) # From actual calibration
#plot_tangential_distortion(ax, -0.007, 0.007) # Artificial values

plt.show()
