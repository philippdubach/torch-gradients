# torch-gradient

Improved and extended version based on [GistNoesis/VisualizeGradient](https://github.com/GistNoesis/VisualizeGradient).

A visualization tool that helps you develop the correct mental picture of what the gradient of a function is. This mental model generalizes beautifully to higher dimensions and is crucial for understanding optimization algorithms. The gradient is one of the most important concepts in calculus and machine learning, but it's often poorly understood. This tool provides an intuitive visual representation that shows:

- **Where gradients live**: In the input space, not along the function curve
- **What gradients point toward**: The direction of steepest increase
- **How optimization works**: Following the arrows to find maxima/minima

You can find alternative gradient visualizations on the [Wikipedia gradient page](https://en.wikipedia.org/wiki/Gradient).

## Features

- **1D Function Visualization**: Plot functions with gradient vectors showing slope and direction
- **2D Function Visualization**: 3D surface plots with gradient vector fields
- **Interactive Plots**: Zoom, rotate, and explore the visualizations
- **Educational Focus**: Clear visual separation between function space and gradient space

## Examples

### 1D Function: f(x) = x³ + 2x² - x

![1D Gradient Visualization]([https://github.com/philippdubach/torch-gradients/blob/5f5d1aca111e70a72f1e4646abbadbed5a90386d/Figure_1.png)

*The blue curve shows the function value. Red arrows show the gradient at each point - note how they live on the x-axis (input space), not along the curve. Arrow length indicates the magnitude of the slope.*

### 2D Function: f(x,y) = sin(√((0.75x)² + y² + 0.01))

![2D Gradient Visualization](path/to/2d_example.png)

*The colored surface shows function values. Black arrows show gradient vectors in the input plane (x-y space), pointing toward the direction of steepest ascent.*

## Quick Start

### Requirements
```bash
pip install torch numpy matplotlib
```

### Usage
```bash
python plotgrad.py
```

The script will generate interactive plots for both example functions. You can:
- Zoom and pan the 1D plot
- Rotate and explore the 3D surface
- Modify the functions in the code to visualize your own examples

### Customizing Functions

Edit `plotgrad.py` to visualize your own functions:

```python
# For 1D functions
def my_function(x):
    return x**2 + 3*x + 1

plot1D(my_function, -5, 5, 0.1, 0.5, 4)

# For 2D functions  
def my_2d_function(x, y):
    return x**2 + y**2  # Simple paraboloid

plot2D(my_2d_function, -3, 3, -3, 3, 0.2, 0.5)
```

## License

MIT License - feel free to use this for educational purposes, research, or your own projects.
