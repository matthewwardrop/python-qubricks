from matplotlib import cm
import matplotlib.pyplot as plt
from mplstyles import cmap as colormap
import numpy as np

def contour_image(x,y,Z,cmap=None,vmax=None,vmin=None,interpolation='nearest',contour_labelsize=9,contour_opts={},imshow_opts={},clegendlabels=[],label=False):
	ax = plt.gca()
	
	x_delta = float((x[-1]-x[0]))/(len(x)-1)/2.
	y_delta = float((y[-1]-y[0]))/(len(y)-1)/2.
	
	extent=(x[0],x[-1],y[0],y[-1])
	
	extent_delta = (x[0]-x_delta,x[-1]+x_delta,y[0]-y_delta,y[-1]+y_delta)
	
	ax.set_xlim(x[0],x[-1])
	ax.set_ylim(y[0],y[-1])
	
	if cmap is None:
		cmap = colormap.reverse(cm.Blues)
	
	Z = Z.transpose()

	#plt.contourf(X,Y,self.pdata,interpolation=interpolation)
	cs = ax.imshow(Z,interpolation=interpolation,origin='lower',aspect='auto',extent=extent_delta,cmap=cmap,vmax=vmax,vmin=vmin, **imshow_opts)

	# Draw contours
	X, Y = np.meshgrid(x, y)
	CS = ax.contour(X, Y, Z, extent=extent, origin='lower', **contour_opts )

	# Label contours
	if label:
		ax.clabel(CS, fontsize=contour_labelsize)

	# Show contours in legend if desired
	if len(clegendlabels) > 0:
		for i in range(len(clegendlabels)):
			CS.collections[i].set_label(clegendlabels[i])
		#ax.legend()
	
	return cs, CS