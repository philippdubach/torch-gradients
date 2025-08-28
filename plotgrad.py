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
    plt.legend()


def plot2D(fun, minx, maxx, miny, maxy, stepf, stepquiver):
    # Create mesh for function evaluation
    XX = np.arange(minx, maxx, stepf)
    YY = np.arange(miny, maxy, stepf)
    X, Y = np.meshgrid(XX, YY)
    
    # Evaluate function
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = fun(th.tensor(X[i, j]), th.tensor(Y[i, j])).item()
    
    # Create mesh for gradient vectors
    XX2 = np.arange(minx, maxx, stepquiver)
    YY2 = np.arange(miny, maxy, stepquiver)
    X2, Y2 = np.meshgrid(XX2, YY2)
    
    # Compute gradients
    U = np.zeros_like(X2)
    V = np.zeros_like(Y2)
    
    for i in range(X2.shape[0]):
        for j in range(X2.shape[1]):
            x_val = th.tensor(X2[i, j], requires_grad=True)
            y_val = th.tensor(Y2[i, j], requires_grad=True)
            z_val = fun(x_val, y_val)
            z_val.backward()
            U[i, j] = x_val.grad.item()
            V[i, j] = y_val.grad.item()
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, 
                          linewidth=0, antialiased=False, alpha=0.6)
    
    # Plot gradient vectors
    Z2 = np.zeros_like(X2)  # Place vectors at z=0 for clarity
    ax.quiver(X2, Y2, Z2, U, V, np.zeros_like(U), 
              color='black', alpha=0.8, length=0.3, normalize=True)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5)
    


def f( x ):
    return x**3 + 2*x**2 -x

plot1D(f,-3,3,0.1,0.5,4)

def g( x,y ):
    R = th.sqrt((0.75*x)**2+y**2 + 0.01)
    Z = th.sin(R)
    return Z

plot2D( g, -5,5.1,-5, 5.1 ,0.25,1.0)
plt.show()