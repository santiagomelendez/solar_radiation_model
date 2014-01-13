import sys
from pylab import show, plot, figure, subplot, mean, genfromtxt, corrcoef, legend

x = genfromtxt(fname=sys.argv[1], skip_header = 1, usecols = 0 )
xi = x

for col in xrange(1,7):
	figure(1)
	subplot(2,3,col )
	y =  genfromtxt(fname=sys.argv[1], skip_header = 1, usecols = col )
	yi = y
	#slope calculated for b = 0 in y = ax + b
	a = mean(xi*yi)/mean(xi**2)
	corrr = corrcoef(yi, a*xi) 
	#print 'corrr ', corrr[1][0]
	print 'slope '+str(col), a
	print corrcoef(yi, a*xi)[1][0] 
	#def func(xi, a):
	#	return a*xi
	#popt, pcov = curve_fit(func, xi, yi)	
	#print 'popt', popt
	#print 'pcov', pcov 
	line = a*xi
	plot(xi,line,'r-',xi,yi,'o', label = a)
	legend()
	#legend(['m ', 'r**2 '],[a, corrr])
show()