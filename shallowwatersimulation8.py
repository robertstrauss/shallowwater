#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import matplotlib as mpl
from matplotlib import animation, rc
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interactive, Button
from IPython.display import display, HTML


# In[ ]:


# useful math functions
def d_dx(a, dx):
    ddx = ( a[:-1] - a[1:] )*(-1/dx) 
    return ddx
def d_dy(a, dy):
    ddy = ( a[:,:-1] - a[:,1:] )*(-1/dy)
    return ddy
def div(u, v, dx, dy):
    div = d_dx(u, dx) + d_dy(v, dy)
    return div

# for generating simple environments or initial conditions
def planegauss(shape, w = 1/2, win=((-2, 2), (-2, 2))):
    h=np.empty(shape, dtype=np.float32)
    npx = np.linspace( win[0][0], win[0][1], shape[0] )
    npy = np.linspace( win[1][0],win[1][1], shape[1] )
    npxx, npyy = np.meshgrid(npx, npy)
    h = np.exp( -np.e * ( npxx*npxx + npyy*npyy ) / (w*w) )
    return (h)
def lingauss(shape, w = 1/2, ax = 0, win = (-2, 2)):
    h=np.empty(shape, dtype=np.float32)
    npx = np.linspace( win[0], win[1], shape[0] )
    npy = np.linspace( win[0], win[1], shape[1] )
    npxx, npyy = np.meshgrid(npy, npx)
    xy = (npyy, npxx)[ax]
    h = np.exp( -np.e * ( xy*xy ) / (w*w) )
    return (h)


# In[ ]:


# physics constants
class p():
    g = np.float32(10.0)

class State():
    g = 10 # m/s/s
    def __init__(self, dx, dy, lat, lon, h, n, u, v):
        
        self.dx = dx
        self.dy = dy
        self.lat = lat 
        self.lon = lon
        self.lat, self.lon = np.meshgrid(self.lat, self.lon) # lattitude/longitude chunk simulation area stretches over
        self.h = h
        
        self.maxws = np.sqrt(np.max(self.h)*p.g) # maximum wave speed
        
        self.n = np.asarray(n, dtype=np.float32) # surface height (eta)
        self.u = np.asarray(u, dtype=np.float32) # x vel array
        self.v = np.asarray(v, dtype=np.float32) # y vel array
        
        #make sure h is the same shap as n (eta)
        assert (np.isscalar(h) or self.h.shape == self.n.shape) # 'or' is short circuit
        
        self.calcDt()
        
        self.coriolis = (np.pi*np.sin(self.lat))/(43200*self.dt) # rotation speed of the earth dtheta/dt
        """ derivation of coriolis force
        U = R*cos(phi)*O
        ui = U+ur
        ur = ui-U
        dU/dphi = -R*sin(phi)*O
        phi = y/R
        dphi/dt = v/R
        dU/dt = v*(-sin(phi)*O)
        dur/dt = dui/dt - dU/dt = v*O*sin(phi)
        dur/dt = v*O*sin(phi)"""
    def calcDt(self, fudge = 5): #calculate optimal value of dt for the height and dx values
        dx = np.min(self.dx)
        dy = np.min(self.dy)
        self.dt = np.min((dx, dy))/(fudge*self.maxws)


class initcons(): # simple set of initial conditions
    size = (100, 100) # grid squares (dx)
    
    dx = np.single(100, dtype=np.float32) # meters
    dy = np.single(100, dtype=np.float32) # meters
    lat = np.linspace(-5, 5, size[0]+1)
    lon = np.linspace(175, 185, size[1]+1)
    
    h = 100-300*planegauss(size, 1)
    n = 1*lingauss(size, 1/4, 0, (-3, 1)) # intial condition
    u = np.zeros((size[0]+1, size[1]+0)) # x vel array
    v = np.zeros((size[0]+0, size[1]+1)) # y vel array

# simple inital environment
state1 = State(initcons.dx, initcons.dy, initcons.lat, initcons.lon, initcons.h, initcons.n, initcons.u, initcons.v)


# In[ ]:


# display functions

def disp3d(aa, box = (None, None, None), lines=(35,35)): # 3d wirefram plot
    # interpert inputs
    xlim = box[0]
    ylim = box[1]
    zlim = box[2]
    if (xlim==None):
        xlim = (0, aa[0].shape[0])
    if (ylim==None): ylim = (0, aa[0].shape[1])
    if (zlim==None):
        ran = np.max(aa[0])-np.min(aa[0])
        zlim = (np.min(aa[0])-ran, np.max(aa[0])+ran)
        zlim = (-2, 2)
    
    #'wires' of the wireframe plot
    x = np.linspace(0, aa[0].shape[0]-1, lines[0], dtype=int)
    y = np.linspace(0, aa[0].shape[1]-1, lines[1], dtype=int)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    #display it
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for a in aa:
        A = a[xx,yy]
        ax.plot_wireframe(xx, yy, A)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    plt.show()

def contour(a): # contour plot
    x = np.arange(0, a.shape[0], dtype=np.float32)
    y = np.arange(0, a.shape[1], dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    #displat it
    plt.figure()
    c = plt.contour(xx, yy, a)
    plt.title("water surface height")
    plt.colorbar()
    return c # output for saving figure

def motioncon(f): # animated height plot, takes in list of 2d height arrays
    #prepare figure/display
    z = f[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(z)
    cb = fig.colorbar(im)
    tx = ax.set_title('water surface height')
    
    def animate(i): # returns i'th element (height array) in f
        im.set_data(f[i])
    
    #display it
    anim = animation.FuncAnimation(fig, animate, frames=len(f))
    plt.show()
    return anim

def vect(u, v, xlim='default', ylim='default', arws=(10, 10), arwsz=100): # vector /motion plot
    #interpert inputs
    if (xlim=='default'): xlim = (0, u.shape[0])
    if (ylim=='default'): ylim = (0, v.shape[1])
    arws = (int(arws[0]), int(arws[1]))
    
    # set up
    x = np.linspace(0, u.shape[0]-1, arws[0], dtype=int)
    y = np.linspace(0, v.shape[1]-1, arws[1], dtype=int)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    uu = u[x,y]
    vv = v[x,y]
    m = np.hypot(uu, vv)
    
    #displat it
    fig, ax = plt.subplots()
    q = ax.quiver(xx, yy, uu, vv, m, scale = 1/arwsz)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()
    
    return q # to be saved later


# In[ ]:


#display initial conditions, tests display functions
contour(state1.n)
# disp3d((state1.n, -state1.h)) # 3d wireframe plot
vect(d_dx(state1.u, state1.dx), d_dy(state1.v, state1.dy), arws=(20, 20))


# In[ ]:


def dndt(h, n, u, v, dx, dy) : # change in n per timestep, defined in diff. equations
    hx = np.empty((n.shape[0]+1, n.shape[1]), dtype=n.dtype) # to be x (u) momentum array
    hy = np.empty((n.shape[0], n.shape[1]+1), dtype=n.dtype)
    hx[1:-1] = ((h+n)[1:] + (h+n)[:-1])/2 # average to same shape as u
    hx[0] = hx[-1] = 0.0 # reflective boundaries/borders
    hy[:,1:-1] = ((h+n)[:,1:] + (h+n)[:,:-1])/2
    hy[:,0] = hy[:,-1] = 0.0
    hx *= u # height/mass->momentum of water column.
    hy *= v
    dndt = (div(hx, hy, -dx, -dy))
    return ( dndt )
def dudt(n, dx) : # change in x vel. (u) per timestep
    dudt = np.empty((n.shape[0]+1, n.shape[1]), dtype=n.dtype) # x accel array
    dudt[1:-1] = d_dx(n, -dx/p.g)
#     dudt += coriolis*v # coriolis force, from earth's rotation
    dudt[0] = dudt[-1] = 0 # reflective boundaries
    return ( dudt )
def dvdt(n, dy) :
    dvdt = np.empty((n.shape[0], n.shape[1]+1), dtype=n.dtype)
    dvdt[:,1:-1] = d_dy(n, -dy/p.g)
#     
    dvdt[:,0] = dvdt[:,-1] = 0
    return ( dvdt )

def land(h, u ,v): # how to handle land/above water area
    #boundaries / land
    coastx = np.less(h, 5) # start a little farther than the coast so H+n is never less than zero
    (u[1:])[coastx] = (u[:-1])[coastx] = 0 # set vel. on either side of land to zero, makes reflective
    (v[:,1:])[coastx] = (v[:,:-1])[coastx] = 0
    return (u, v)

def forward(h, n, u, v, dt, dx, dy, doland, beta=0): # forward euler and forward/backward timestep
    # beta = 0 => forward, beta = 1 => forward-backward
    n1 = n + ( dndt(h, n, u, v, dx, dy) )*dt
    u1 = u + ( beta*dudt(n1, dx) +  (1-beta)*dudt(n, dx) )*dt
    v1 = v + ( beta*dvdt(n1, dy) +  (1-beta)*dvdt(n, dy) )*dt
    u1, v1 = doland(h, u1, v1) # handle any land in the simulation
    return n1, u1, v1

def fbfeedback(h, n, u, v, dt, dx, dy, doland, beta=1/3, eps=2/3): # forward backward feedback timestep
    n1g, u1g, v1g = forward(h, n, u, v, dt, dx, dy, doland, beta) # forward-backward first guess
    #feedback on guess
    n1 = n + 0.5*(dndt(h, n1g, u1g, v1g, dx, dy) + dndt(h, n, u, v, dx, dy))*dt
    u1 = u + 0.5*(eps*dudt(n1, dx)+(1-eps)*dudt(n1g, dx)+dudt(n, dx))*dt
    v1 = v + 0.5*(eps*dvdt(n1, dy)+(1-eps)*dvdt(n1g, dy)+dvdt(n, dy))*dt
    u1, v1 = doland(h, u1, v1) # how to handle land/coast
    return n1, u1, v1

def timestep(h, n, u, v, dt, dx, dy): return fbfeedback(h, n, u, v, dt, dx, dy, land) # switch which integrator/timestep is in use

#


# In[ ]:


def simulate(state, t): # gives surface height array of the system after evert dt
    h, n, u, v, dx, dy, dt = state.h, state.n, state.u, state.v, state.dx, state.dy, state.dt
    if (dt <= 0):# dont try if timstep is zero or negative
        return False
    f=[n] # list of height arrays over time
    itr = 0
    while (itr < t):# iterate t times with interval size dt
        n, u, v = timestep(h, n, u, v, dt, dx, dy) # pushes n, u, v one step into the future
        f.append(n) # add new surface height to array
        itr += dt
    return f # return surface height array time list


# In[ ]:


f = simulate(state1, 250)[::5] # simulate a system for 250 seconds, only taking every fifth frame/dt image
motioncon(f) # displat as an animation


# In[ ]:


#wavespeed and differential tests
import unittest
fooo = []
class testWaveSpeed(unittest.TestCase): # tests if the wave speed is correct
    def setUp(self):
        self.dur = 500 # duration of period to calculate speed over
        self.size = (10, 1000) # grid squares (dx's)
        self.dx = np.single(100, dtype=np.float32) # meters
        self.dy = np.single(100, dtype=np.float32)
        self.lat = np.linspace(0, 0, self.size[0]) # physical location the simulation is over
        self.lon = np.linspace(0, 0 , self.size[1])
        self.h = 100
        self.n = 1*lingauss(self.size, 1/4, 1) # intial condition single wave in the center
        self.u = np.zeros((self.size[0]+1, self.size[1]+0)) # x vel array
        self.v = np.zeros((self.size[0]+0, self.size[1]+1)) # y vel array
        self.margin = 0.01 # error margin of test
    def calcWaveSpeed(self, ar1, ar2, Dt): # calculat how fast the wave is propagating out
        midstrip1 = ar1[int(ar1.shape[0]/2),int(ar1.shape[1]/2):]
        midstrip2 = ar2[int(ar1.shape[0]/2),int(ar2.shape[1]/2):]
        peakloc1 = np.argmax(midstrip1)
        peakloc2 = np.argmax(midstrip2)
        plt.figure()
        plt.clf()
        plt.plot(midstrip1)
        plt.plot(midstrip2)
        plt.show()
        speed = (peakloc2 - peakloc1)*self.dy/Dt
        return speed
    def calcExactWaveSpeed(self): # approximently how fast the wave should be propagating outwards
        ws = np.sqrt(p.g*np.average(self.h))
        return ws
    def test_wavespeed(self): # test if the expected and calculated wave speeds line up approcimently
        self.testStart = State(self.dx, self.dy, self.lat, self.lon, self.h, self.n, self.u, self.v)
        self.testEndN = simulate(self.testStart, self.dur)[-1]
        calcedws = self.calcWaveSpeed( self.testStart.n, self.testEndN, self.dur )
        exactws = self.calcExactWaveSpeed()
        err = (calcedws - exactws)/exactws
        print(err, self.margin)
        assert(abs(err) < self.margin) # error margin
    def tearDown(self):
        del(self.dur)
        del(self.dx)
        del(self.dy)
        del(self.lat)
        del(self.lon)
        del(self.size)
        del(self.h)
        del(self.n)
        del(self.u)
        del(self.v)

class testdifferential(unittest.TestCase): # differental function test (d_dx)
    def setUp(self):
        self.a = np.arange(144) # test input
        self.a = self.a.reshape(12, 12) # make into 2d array
        self.ddthreshold = 1E-16
    def test_ddx(self):
        da = d_dx(self.a, 1)
        diff = np.abs(da[1:-1] - np.mean(da[1:-1]))
        maxdiff = np.max(diff)
        self.assertTrue(np.all(np.abs(da[-1:1] < self.ddthreshold)),"expected zero along borders")
        self.assertTrue(np.all(diff < self.ddthreshold),"Expected constant d_dx less than %f but got %f"%(self.ddthreshold,maxdiff))
    def tearDown(self):
        del(self.a)
        del(self.ddthreshold)

unittest.main(argv=['first-arg-is-ignored'], exit=False)
#You can pass further arguments in the argv list, e.g.
#unittest.main(argv=['ignored', '-v'], exit=False)      
#unittest.main()


# In[ ]:


import timeit

def foo():
    simulate(state1, 250)

print('time to simulate:', timeit.timeit(foo, number=8)) # time how long it takes to simulate 250 frames

