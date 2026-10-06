## Serie de ejemplos de uso de SYMPY para señales y sistemas

En el folder de `1_Notebooks` se encuentra ejemplos de uso de una librería con funciones para señales y sistemas, similar a MATLAB.

```python
from isb_lib import ISB, plot_pz, pzmap
X_s = 1 / (s**2 + 2*s + 5)
plot_pz(X_s)

from sympy import symbols, DiracDelta, Heaviside
# 3.1 Ejemplo: Escalera (Step)
x_t = Heaviside(t)
X_s = 1/s
t_vals = np.linspace(-1, 5, 400)
x_vals = [x_t.subs(t, val) for val in t_vals]
X_vals = [X_s.subs(s, 1j*omega) for omega in np.linspace(-5, 5, 200)]
# ...
```

En el folder de `2_Images` se encuentran imágenes de ejemplos de uso de la librería.