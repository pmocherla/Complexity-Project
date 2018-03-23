# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:04:11 2017

@author: Priyanka Mocherla
@version: 2.7

This code contains functions to simulate the Oslo model for varying system sizes. The code can also collapse the data for different systems and extract key features from each of the models.

Also includes the code required to complete the project tasks.

Functions in this module:
    - oslo_init
    - slope_reboot
    - oslo_drive
    - oslo_relax
    - oslo
    - smoothing
    - average_height
    - stand_dev
    - data_prob
    - cutoff time
    - scaling_correction
    - stand_dev
    - cut_off_theoreticalpref
    - cut_off_theoreticalrand

"""


import numpy as np
import random as ran
import matplotlib.pyplot as plt
from log_bin import *

def oslo_init(L,p,grains):
    """ 
       returns the initialised arrays of heights, slopes, slope threshold and avalanches.
        
        Args: 
        L: integer, the size of the system
        p: float, the slope threshold probability
        grains: integer, the number of grains run through the system                  
    """
    heights = np.zeros(grains)
    slope = np.zeros(L)
    slopethresh = np.zeros(L)
    avalanche = np.zeros(grains)
    for i in range(L):
        trial = ran.random()
        if trial <= p:
            slopethresh[i] = 1
        else:
            slopethresh[i] = 2 #sets the values of the threshold based on the threshold probability
    return heights, slope, slopethresh, avalanche
    
def slope_reboot(slopethreshi, p):
    """ 
       returns a new slope threshold value upon relaxation of the i-th site.
        
        Args: 
        slopethreshi: integer, the i-th element of the slope threshold array
        p: float, the slope threshold probability              
    """
    trial  = ran.random()
    if trial <=p:
        slopethreshi = 1
    else:
        slopethreshi = 2
    return slopethreshi
        
        
def oslo_drive(slope):
    """ 
       returns the new values of the slope after being driven by an incoming grain.
        
        Args: 
        slope: array eg. np.array([x1,x2,x3...])
    """
    slope[0] = slope[0]+1
    return slope
    
def oslo_relax(slope, slopethresh,L,p,avalanche):
    """ 
        returns the new values of the slope, slope threshold and number of avalanches after the system relaxes once.
        
        Args: 
        slope: array eg. np.array([x1,x2,x3...])
        slopethresh: array eg. np.array([x1,x2,x3...])
        L: integer, the size of the system
        p: float, the slope threshold probability
        avalanche: integer, the number relaxations that occur in one loop i.e. the number of avalanches.   
    """
    for i in range(L):
        if slope[i] > slopethresh[i]:
            if i == 0:
                slope[0] = slope[0]-2
                slope[1] = slope[1] +1
                slopethresh[i] = slope_reboot(slopethresh[i],p)
                avalanche = avalanche + 1
            elif i == (L-1):
                slope[L-1] = slope[L-1]-1
                slope[L-2] = slope[L-2] +1
                slopethresh[i] = slope_reboot(slopethresh[i],p)
                avalanche = avalanche + 1
            else:
                slope[i] = slope[i]-2
                slope[i+1] = slope[i+1] +1
                slope[i-1] = slope[i-1] +1
                slopethresh[i] = slope_reboot(slopethresh[i],p)
                avalanche = avalanche + 1
    return slope, slopethresh,avalanche


def oslo(L,p,grains):
    """ 
       returns the final values of the heights, slopes, slope thresholds at each site and avalanche size upon addition of each grain.
        
        Args: 
        L: integer, the size of the system
        p: float, the slope threshold probability
        avalanche: integer, the number relaxations that occur in one loop i.e. the number of avalanches.  
    """
    h,z,zth,s = oslo_init(L,p,grains)
    for j in range(grains):
        z = oslo_drive(z)
        while np.any((zth-z)<0):
            z,zth,s[j] = oslo_relax(z,zth,L,p,s[j])
        h[j] = sum(z)
    return h,z,zth,s
    
def smoothing(heights,w):
    """ 
       returns the a moving average of an input array and a reset number of grains.
        
        Args: 
        heights: array eg. np.array([x1,x2,x3...])
        w: integer, the smoothing factor of the data
    """
    smoothed = np.zeros(heights.size-2*w)
    for i in range(smoothed.size):
        smoothed[i] = (1/(2.*w+1))*np.sum(heights[i:2*w+1+i])
    grains = np.arange(w,(heights.size-w))
    return smoothed,grains
    
def average_height(heights):
    """ 
       returns the average height of an input array.
        
        Args: 
        heights: array eg. np.array([x1,x2,x3...])
    """
    return np.sum(heights)/heights.size
    
def stand_dev(heights):
    """ 
       returns the standard deviation of an input array.
        
        Args: 
        heights: array eg. np.array([x1,x2,x3...])
    """
    return np.sqrt(np.sum(heights*heights)/heights.size-(np.sum(heights)/heights.size)*(np.sum(heights)/heights.size))
    
def data_prob(data):
    """ 
       returns a dictionary of data keys and corresponding frequency probability.
        
        Args: 
        data: array eg. np.array([x1,x2,x3...])
    """
    dataprob = {}
    for datum in data:
        if datum not in dataprob:
            dataprob[datum] = 0
        dataprob[datum] = dataprob[datum] + 1./data.size
    return dataprob
    
def cutofftime(L,m,b,h):
    """ 
       returns the cut-off time for each system size by finding the point of intersection of the fitted transient line and average height.
        
        Args: 
        L: array eg. np.array([x1,x2,x3...]), system sizes
        m: array eg. np.array([x1,x2,x3...]), the gradient of the fitted transient region
        b: array eg. np.array([x1,x2,x3...]), intercept of the fitted transient region
        h: array eg. np.array([x1,x2,x3...]), average heights for each corresponding system size
    """
    cut_off = np.zeros(L.size)
    for i in range(L.size):
        cut_off[i] = (np.exp(-b[i])*h[i])**(1/m[i])
    return L,cut_off
    
def scaling_correction(start,stop,increment,L,h):
    """ 
       returns the value of a0 and w1 from the correction to scaling formula. optimises the value of a0 which returns the smallest residual value
        
        Args: 
        start: float, optimisation start value
        stop: float, optimisation end value
        increment: float, optimisation step size
        L: array eg. np.array([x1,x2,x3...]), system sizes
        h: array eg. np.array([x1,x2,x3...]), average heights for each corresponding system size
    """
    test = np.polyfit(np.log(L),np.log(start-h/L),1, full = True)[1]
    for a0 in np.arange(start+increment,stop,increment):
        m,b = np.polyfit(np.log(L),np.log(a0-h/L),1)
        if np.polyfit(np.log(L),np.log(a0-h/L),1, full = True)[1] < test:
            test = np.polyfit(np.log(L),np.log(a0-h/L),1, full = True)[1]
            continue
        else:
            return a0, -1*m
    
    

#---------------------------------- Project Tasks ------------------------------------#
#Variable Data
grains = 580000 #number of grains to add to the system. Note: 1mill grains ~ 2 hour runtime
start = 80000 #start of steady state regime. If max L = 256, reccomended start time t > 60000. For L = 512, t > 250000
w = 100 #Data smoothing factor
a = 1.4 #Log bin multiplication factor

#DATA FOR PLOTTING - DO NOT CHANGE
L2 = np.array([16,32,64,128,256,512])
m = np.array([0.56379806,0.54469388, 0.52788457,0.51621403,0.51024593,0.50634806])
b = np.array([0.20303031,0.28378633, 0.37141845,0.45312685,0.50175931,0.53902392])
htwo = np.array([26.5,53.8,109.1,219.4, 440.5,883.1])
L = np.array([8,16,32,64,128,256,512])
h = np.array([13.04, 26.52833333,53.82333333,109.0581333,219.3535333,440.497125,883.12156])
h_uncert = np.array([0.960768097,1.13634456,1.340240532,1.573015293,1.882134786,2.20428962, 2.623248803])
cutoff = np.array([126,	687,	2648,	14259,	54829,	294657])
systems = np.array([8,16,32,64,128,256])
approx = 200000
h0,h1,h2,h3,h4,h5,h8 = oslo(16,0.5,grains)[0],oslo(32,0.5,grains)[0],oslo(64,0.5,grains)[0],oslo(128,0.5,grains)[0],oslo(256,0.5,grains)[0],oslo(512,0.5,grains)[0],oslo(8,0.5,grains)[0]
s0,s1,s2,s3,s4,s8 = oslo(16,0.5,grains)[3][start:grains],oslo(32,0.5,grains)[3][start:grains],oslo(64,0.5,grains)[3][start:grains],oslo(128,0.5,grains)[3][start:grains],oslo(256,0.5,grains)[3][start:grains],oslo(8,0.5,grains)[3][start:grains]
kmom = np.array([1,2,3,4,5])
grad = np.array([1.000874623,3.192612089,5.429626879,7.676656264,9.925195027])
t = np.arange(grains)
plt.rcParams["font.family"] = "Courier New, monospace"
#-------------------------------------------------------------------------------
#################################TASKS#####################################
"""
#Task 1: Testing the Oslo Model

#Testing average heights and avalanche sizes at steady state

ha,za,ztha,sa = oslo(16,1,10000)#Initialising Oslo data
hb,zb,zthb,sb = oslo(32,1,10000)
hc,zc,zthc,sc = oslo(16,0.5,10000)
hd,zd,zthd,sd = oslo(32,0.5,10000)
    
print 'For p = 1, L = 16, <h> is ' +str(average_height(ha[3000:10000])) + ' and <s> is: '+str(average_height(sa[3000:10000]))
print 'For p = 1, L = 32, <h> is ' +str(average_height(hb[3000:10000]))+ ' and <s> is: '+str(average_height(sb[3000:10000]))
print 'For p = 0.5, L = 16, <h> is ' +str(average_height(hc[3000:10000]))+ ' and <s> is: '+str(average_height(sc[3000:10000]))
print 'For p = 0.5, L = 32, <h> is ' +str(average_height(hd[3000:10000]))+ ' and <s> is: '+str(average_height(sd[3000:10000]))
"""

#------------------------------------------------------------------------------
"""
#Task 2A: Height as function of time

plt.plot(t,h0, '-', label = 'L = 16')
plt.plot(t,h1, '-', label = 'L = 32')
plt.plot(t,h2, '-', label = 'L = 64')
plt.plot(t,h3, '-', label = 'L = 128')
plt.plot(t,h4, '-', label = 'L = 256')
plt.plot(t,h5, '-', label = 'L = 512')

#Fitting the transient region
j,k = np.polyfit(np.log(t)[1:approx],np.log(h5)[1:approx],1)
plt.plot(t,np.exp(np.log(t)*j+k),lw = 1.25,color = 'black')
print 'Transient ~ t^' +str(j)

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#plt.title("System Height")
plt.xlabel('$t$', fontsize = 20)
plt.ylabel('$h$', fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend(loc = 2)
plt.show()
"""
#------------------------------------------------------------------------------
"""
#Task 2A: Plotting the cut-off time, average height with system size
x,y = np.polyfit(np.log(cutofftime(L2,m,b,h2)[0]),np.log(cutofftime(L2,m,b,h2)[1]),1) # Calculating the gradient of the line to find power law between t_c and L
print "t_c ~ L^" +str(x)

p,q = np.polyfit(np.log(L2),np.log(htwo),1) # Calculating the gradient of the line to find power law between <h> and L
print "<h> ~ L^" +str(p)

#t_c vs L
plt.figure()
plt.loglog(cutofftime(L2,m,b,h2)[0],cutofftime(L2,m,b,h2)[1],'o',color = "#22BDE6") # plotting cut-off data
plt.loglog(L2,np.exp(x*np.log(L2)+y),color = "#FF8200") #plotting fitted line
plt.title("Variation of cut-off time with system size")
plt.xlabel("$L$", fontsize = 22)
plt.ylabel("$t_c$", fontsize = 22)

#<h> vs L
plt.figure()
plt.plot(L2,htwo,'o',color = "#22BDE6") # plotting cut-off data
plt.plot(L2,np.exp(p*np.log(L2)+q),color = "#FF8200") #plotting fitted line
plt.title("Variation of average height time with system size")
plt.xlabel("$L$", fontsize = 22)
plt.ylabel("$<h>$", fontsize = 22)

plt.show()
"""
#-------------------------------------------------------------------------------
"""
#Task 2B: Data Collapse

plt.plot(smoothing(h0,w)[1]/16**2.0,smoothing(h0,w)[0]/16, '-', label = 'L = 16')
plt.plot(smoothing(h1,w)[1]/32**2.0,smoothing(h1,w)[0]/32, '-', label = 'L = 32')
plt.plot(smoothing(h2,w)[1]/64**2.0,smoothing(h2,w)[0]/64, '-', label = 'L = 64')
plt.plot(smoothing(h3,w)[1]/128**2.0,smoothing(h3,w)[0]/128, '-', label = 'L = 128')
plt.plot(smoothing(h4,w)[1]/256**2.0,smoothing(h4,w)[0]/256, '-', label = 'L = 256')
plt.plot(smoothing(h5,w)[1]/512**2.0,smoothing(h5,w)[0]/512, '-', label = 'L = 512')

plt.title("System Height Data Collapse")
plt.xlabel('$t/ t_c $', fontsize = 22)
plt.ylabel('$h / h_c $', fontsize = 22)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend(loc = 4)
plt.show()
"""
#-----------------------------------------------------------------------------
"""
#Task 2C: Average height, standard deviation with system size and correction to scaling

d,e = np.polyfit(np.log(L),np.log(h),1) # Calculating the gradient of the line to find power law between <h> and L
print "<h> ~ L^" +str(d)
#<h> vs L
plt.figure()
plt.title("Variation of average height with system size") 
plt.plot(L,h,'o',color = "#22BDE6") # data
plt.plot(L,np.exp(d*np.log(L)+e),color = "#FF8200") #fitted line
plt.xlabel('$L$', fontsize = 22)
plt.ylabel('$<h>$', fontsize = 22)

#<h>/L vs L (to emphasise correction to scaling effects)
p,q = np.polyfit(L,h,1) # Calculating the gradient between <h> and L
plt.figure()
plt.title('Corrections to scaling - Average Height')
plt.plot(L,h/L,'o',color= "#22BDE6")#Data
plt.plot(L,p*L/L,'-',color = "#FF8200",lw=1.5)#fitted line
plt.xlabel('$L$', fontsize = 22)
plt.ylabel('$<h>/L$', fontsize = 22)


#Corrections to scaling - Average height
f,g = np.polyfit(L,h,1) # Using the correction to scaling formula to estimate the value of a0
print "a0 estimate: " +str(f)
print "a0 optimised: " +str(scaling_correction(1.725,1.735,0.0001,L,h)[0])    
print "w1 optimised: " +str(scaling_correction(1.725,1.735,0.0001,L,h)[1]) 

a,c = np.polyfit(np.log(L),np.log(h_uncert),1) # Calculating the gradient of the line to find power law between sigma_h and L
print "sigma_h ~ L^" +str(a)


#sigma_h vs L
plt.figure()
plt.title("Variation of height standard deviation with system size")
plt.plot(L,np.exp((a*np.log(L)+c)),color = "#FF8200")#fitted line
plt.loglog(L,h_uncert,'o',color = "#22BDE6")#data
plt.xlabel("$L$", fontsize = 22)
plt.ylabel("$\sigma_h$", fontsize = 22) 


#Corrections to scaling emphasis - standard deviation
plt.figure()
plt.title('Corrections to scaling - Standard Deviation')
plt.loglog(L,h_uncert/L**a,'o',color= "#22BDE6")
n,o = np.polyfit(np.log(L),np.log(h_uncert/L**a),1)#Refitting scaled data
plt.plot(L,np.exp(np.log(L)*n+o),color = "#FF8200",lw = 1.5)
plt.xlabel("$L$", fontsize = 20)
plt.ylabel('$\sigma_h / L^{0.24}$', fontsize = 20)

plt.show()
"""
#-------------------------------------------------------------------------------
"""
#Task 2D: Height Probabilities and Data Collapse

#Generating the probability distributions
l0,l1,l2,l3,l4,l5,l8 = sorted(data_prob(h0[start:grains]).items()),sorted(data_prob(h1[start:grains]).items()),sorted(data_prob(h2[start:grains]).items()),sorted(data_prob(h3[start:grains]).items()),sorted(data_prob(h4[start:grains]).items()),sorted(data_prob(h5[start:grains]).items()),sorted(data_prob(h8[start:grains]).items())

#Sorting the probabilities into coordinate sets
x0, y0 = np.array(zip(*l0))
x1, y1 = np.array(zip(*l1))
x2, y2 = np.array(zip(*l2))
x3, y3 = np.array(zip(*l3))
x4, y4 = np.array(zip(*l4))
x5, y5 = np.array(zip(*l5))
x8, y8 = np.array(zip(*l8))

#P(h,L) vs h
plt.figure()
#plt.title('Height Probability Distributions')
plt.plot(x8,y8,label = 'L = 8')
plt.plot(x0,y0,label = 'L = 16')
plt.plot(x1,y1,label = 'L = 32')
plt.plot(x2,y2,label = 'L = 64')
plt.plot(x3,y3,label = 'L = 128')
plt.plot(x4,y4,label = 'L = 256')
plt.plot(x5,y5,label = 'L = 512')
plt.xlabel('$h$', fontsize = 18)
plt.ylabel('$P (h ; L)$', fontsize = 18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend(loc=1)


#Data Collapse
plt.figure()
#plt.title('Height Probability Data Collapse')
plt.plot((x8-h[0])/h_uncert[0],y8*h_uncert[0],label = 'L = 8')
plt.plot((x0-h[1])/h_uncert[1],y0*h_uncert[1],label = 'L = 16')
plt.plot((x1-h[2])/h_uncert[2],y1*h_uncert[2],label = 'L = 32')
plt.plot((x2-h[3])/h_uncert[3],y2*h_uncert[3],label = 'L = 64')
plt.plot((x3-h[4])/h_uncert[4],y3*h_uncert[4],label = 'L = 128')
plt.plot((x4-h[5])/h_uncert[5],y4*h_uncert[5],label = 'L = 256')
plt.plot((x5-h[6])/h_uncert[6],y5*h_uncert[6],label = 'L = 512')
#plt.xscale('log')
#plt.yscale('log')
plt.xlabel('$(h-<h>)/\sigma_h$', fontsize = 18)
plt.ylabel('$\sigma_h P (h ; L)$', fontsize = 18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend(loc=1)

plt.show()
"""
#------------------------------------------------------------------------------
"""
#Task 3A: Testing Logbinning parameters
N = np.array([10000,100000,1000000])

#Plots the avalanche sizes for various N
for number in N:
    plt.figure()
    plt.title('a = '+str(a)+' with N = '+str(number))     
    s = oslo(256,0.5,grains)[3][start:number+start]
    lists = sorted(data_prob(s).items())
    x, y = zip(*lists)
    b, c = log_bin(s, 1.,1.5, a, debug_mode=True, drop_zeros=False)
    plt.loglog(x, y, 'x',color = "#22BDE6")
    plt.loglog(b, c, '-',color = "#FF8200")
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.xlabel('$s$', fontsize = 20)
    plt.ylabel('$P (s ; L)$', fontsize = 20)
    plt.show() 
"""
#------------------------------------------------------------------------------
"""
#Task 3B and 3C: Plotting Avalanche sizes and Data Collapse Exponents

#Binning and sorting data
b0,c0 = log_bin(s0,1,1.5,a,debug_mode=True)
b1,c1 = log_bin(s1,1,1.5,a,debug_mode=True)
b2,c2 = log_bin(s2,1,1.5,a,debug_mode=True)
b3,c3 = log_bin(s3,1,1.5,a,debug_mode=True)
b4,c4 = log_bin(s4,1,1.5,a,debug_mode=True)
b8,c8 = log_bin(s8,1,1.5,a,debug_mode=True)

#Plotting data
plt.figure()
plt.loglog(b8, c8, '-',label = 'L = 8')
plt.loglog(b0, c0, '-',label = 'L = 16')
plt.loglog(b1, c1, '-',label = 'L = 32')
plt.loglog(b2, c2, '-',label = 'L = 64')
plt.loglog(b3, c3, '-',label = 'L = 128')
plt.loglog(b4, c4, '-',label = 'L = 256')
plt.xlabel('$s$', fontsize = 20)
plt.ylabel('$P (s ; L)$', fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend(loc=3)

#Estimating the power law decay constant
r,s = np.polyfit(np.log(b4[4:23]),np.log(c4[4:23]),1)
plt.plot(b4,np.exp(np.log(b4)*r+s),'-',color = "black",lw = 1.5)
print 'tau_s ~ ' +str(r)

#Estimating D cut off value
#cut off avalanche vs L
plt.figure()
plt.title('Cut off avalanche size with system size')
plt.loglog(systems,cutoff,'o',color = "#22BDE6")
t1,t2 = np.polyfit(np.log(systems), np.log(cutoff),1)#Fitting data
plt.plot(systems, np.exp(np.log(systems)*t1+t2),'-',color = "#FF8200")
print 'D ~ ' +str(t1)
plt.xlabel('$L$', fontsize = 20)
plt.ylabel('$s_c$', fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

#Plotting Data Collapse
plt.figure()
plt.title('Avalanche size data collapse')
plt.loglog(b8/8**t1, c8*(np.array(b8))**-r, '-',label = 'L = 8')
plt.loglog(b0/16**t1, c0*(np.array(b0))**-r, '-',label = 'L = 16')
plt.loglog(b1/32**t1, c1*(np.array(b1))**-r, '-',label = 'L = 32')
plt.loglog(b2/64**t1, c2*(np.array(b2))**-r, '-',label = 'L = 64')
plt.loglog(b3/128**t1, c3*(np.array(b3))**-r, '-',label = 'L = 128')
plt.loglog(b4/256**t1, c4*(np.array(b4))**-r, '-',label = 'L = 256')

plt.xlabel('$s/L^D$', fontsize = 20)
plt.ylabel(r'$s^{-\tau_s}P (s ; L)$', fontsize = 20)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend(loc=3)

plt.show() 
"""
#-------------------------------------------------------------------------------
"""
#Task 3D: Moment Analysis
#Finding the kth moment of the avalanches of each system
plt.figure()
plt.title('k-th moment data')
for k in np.arange(1,6):
    plt.loglog(8,np.sum(s8**k)/float(grains-start),'o',color= "#22BDE6")
    plt.loglog(16,np.sum(s0**k)/float(grains-start),'o',color= "#22BDE6")
    plt.loglog(32,np.sum(s1**k)/float(grains-start),'o',color= "#22BDE6")
    plt.loglog(64,np.sum(s2**k)/float(grains-start),'o',color= "#22BDE6")
    plt.loglog(128,np.sum(s3**k)/float(grains-start),'o',color= "#22BDE6")
    plt.loglog(256,np.sum(s4**k)/float(grains-start),'o',color= "#22BDE6")
    m,b = np.polyfit(np.log(systems)[2::],np.log(np.array([np.sum(s8**k)/float(grains-start),np.sum(s0**k)/float(grains-start),np.sum(s1**k)/float(grains-start),np.sum(s2**k)/float(grains-start),np.sum(s3**k)/float(grains-start),np.sum(s4**k)/float(grains-start)]))[2::],1)
    plt.plot(systems,np.exp(m*np.log(systems)+b),color = "#FF8200")
    print 'Moment: ' + str(k) + ", grad: "+str(m)

plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.xlabel('$L$', fontsize = 18)
plt.ylabel('$<s^k>$', fontsize = 18) 

#Plotting  the kth moment against the system size exponent
plt.figure()
plt.title('D and tau_s extraction')
p1,p2 = np.polyfit(kmom,grad,1)
plt.plot(kmom,p1*kmom+p2,color = "#FF8200")
print 'D ~ ' +str(p1)
print 'tau_s ~ ' +str(1-p2/p1)

plt.plot(kmom,grad,'o',color= "#22BDE6")
plt.xlabel('$k$', fontsize = 18)
plt.ylabel(r'$D(1+k-\tau_s)$', fontsize = 18)   
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
      
plt.show()
"""