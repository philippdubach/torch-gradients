import numpy as np
import matplotlib.pyplot as plt
import torch as th
from matplotlib import cm


def plot1D(fun, minx, maxx, stepf, stepquiver, clipq):
    # Create gradient function
    def gf(x):
        x_tensor = th.tensor(x, requires_grad=True)
        y = fun(x_tensor)
        y.backward()
        return x_tensor.grad
    
    # Plot function
    x = np.arange(minx, maxx, stepf)
    y = [fun(th.tensor(xi)).item() for xi in x]
    
    # Plot gradient vectors
    xx = np.arange(minx, maxx, stepquiver)
    x0 = xx
    y0 = np.zeros_like(xx)
    v = np.zeros_like(xx)
    u = np.array([th.clip(gf(xi), -clipq, clipq).item() for xi in xx])
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='Function')
    plt.quiver(x0, y0, u, v, angles='xy', scale_units='xy', scale=1, color='red', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('1D Function with Gradient Vectors')
    plt.legend()




def plot2D( fun, minx,maxx, miny,maxy,stepf,stepquiver ):
    XX = np.arange(minx, maxx, stepf)
    YY = np.arange(miny, maxy, stepf)
    X, Y = np.meshgrid(XX, YY)

    Z = fun( th.tensor(X),th.tensor(Y) ).numpy()

    XX2 = np.arange(minx, maxx, stepquiver)
    YY2 = np.arange(miny, maxy, stepquiver)
    gradg = th.func.grad(fun,(0,1))

    gg = [ gradg(th.tensor(xi),th.tensor(yi)) for yi in YY2 for xi in XX2 ]

    rshpgg = np.reshape( np.array( [ ( x.numpy(),y.numpy()) for x,y in gg]), (YY2.shape[0],XX2.shape[0],2))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # Plot the surface.


    x, y, z = np.meshgrid(XX2,
                        YY2,
                        0)
    u = rshpgg[:,:,0:1]
    v = rshpgg[:,:,1:2]
    w = 0

    


    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False, alpha=0.5)
    ax.quiver(x,y,z,u,v,w, color = 'black')
    


def f( x ):
    return x**3 + 2*x**2 -x

plot1D(f,-3,3,0.1,0.5,4)

def g( x,y ):
    R = th.sqrt((0.75*x)**2+y**2 + 0.01)
    Z = th.sin(R)
    return Z

plot2D( g, -5,5.1,-5, 5.1 ,0.25,1.0)
plt.show()