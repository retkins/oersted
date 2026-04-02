# Magnetized Element Calculation

Equations 60 and 61 of [CRYO-06-034](https://supermagnet.sourceforge.io/notes/CRYO-06-034.pdf) 
describe how to calculate the field from a magnetized polyhedron at a field point in 3D 
space. These functions involve the gradient of a function of the edge integrals of the polyhedron,
which is messy. 

To compute these gradients, the use of sympy was employed:

```python
from sympy import *

x, y, z = symbols('x y z', real=True)
r = sqrt(x**2 + y**2 + z**2)
E = ln(x + r) + z/y * (atan(x*z/(y*r)) - atan(x/y))

dE_dx = simplify(diff(E, x))
dE_dy = simplify(diff(E, y))
dE_dz = simplify(diff(E, z))
```

This produces:
```python
dE_dx = (x**2 + y**2 + z**2 - z*sqrt(x**2 + y**2 + z**2))/((x**2 + y**2)*sqrt(x**2 + y**2 + z**2))  

dE_dy = y/((x + sqrt(x**2 + y**2 + z**2))*sqrt(x**2 + y**2 + z**2)) + z*(x/(y**2*(x**2/y**2 + 1)) - (x*z/(x**2 + y**2 + z**2)**(3/2) + x*z/(y**2*sqrt(x**2 + y**2 + z**2)))/(x**2*z**2/(y**2*(x**2 + y**2 + z**2)) + 1))/y + z*(atan(x/y) - atan(x*z/(y*sqrt(x**2 + y**2 + z**2))))/y**2  

dE_z = (-y**2*atan(x/y) + y**2*atan(x*z/(y*sqrt(x**2 + y**2 + z**2))) + y*z - z**2*atan(x/y) + z**2*atan(x*z/(y*sqrt(x**2 + y**2 + z**2))))/(y*(y**2 + z**2))
```
 
And can be somewhat simplified to:
(in progress)
```python
r = sqrt(x**2 + y**2 + z**2)
r2 = x**2 + y**2 + z**2

dE_dx = (r2 - z*r)/((x**2 + y**2)*r)  

dE_dy = y/((x + r)*r) + z*(x/(y**2*(x**2/y**2 + 1)) - (x*z/(r2)**(3/2) + x*z/(y**2*r))/(x**2*z**2/(y**2*(r2)) + 1))/y + z*(atan(x/y) - atan(x*z/(y*r)))/y**2  

dE_z = (-y**2*atan(x/y) + y**2*atan(x*z/(y*r)) + y*z - z**2*atan(x/y) + z**2*atan(x*z/(y*r)))/(y*(y**2 + z**2))
```
