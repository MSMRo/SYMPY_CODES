import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

class ISB:
    """
    Librería ISB (Ingeniería de Señales y Búsqueda de Sistemas)
    Diseñada para automatizar el análisis simbólico y gráfico de señales
    y sistemas en Python de forma equivalente a MATLAB.
    """

    @staticmethod
    def plot_pz(expr, t_range=(0, 10), num_pts=500, var_t='t', var_s='s', show_unit_circle=True, figsize=(12, 5)):
        """
        Muestra en paralelo la gráfica en el tiempo x(t) y el Diagrama de Polos y Ceros en el plano s.

        Parámetros:
        -----------
        expr : sympy.Expr
            Expresión simbólica en tiempo (t) o en Laplace (s).
        t_range : tuple, opcional (por defecto: (0, 10))
            Rango de tiempo (t_min, t_max) para graficar en el tiempo.
        num_pts : int, opcional (por defecto: 500)
            Número de puntos de evaluación numérica.
        var_t : str, opcional (por defecto: 't')
            Nombre de la variable simbólica de tiempo.
        var_s : str, opcional (por defecto: 's')
            Nombre de la variable simbólica de Laplace.
        show_unit_circle : bool, opcional (por defecto: True)
            Dibuja el círculo unitario |s| = 1 como referencia.
        figsize : tuple, opcional (por defecto: (12, 5))
            Tamaño de la figura.

        Retorna:
        --------
        tuple: (X_s, x_t, polos, ceros)
        """
        # Convertir objetos TransferFunction a expresiones simbólicas de SymPy si aplica
        if hasattr(expr, "to_expr"):
            expr = expr.to_expr()

        if not isinstance(expr, sp.Basic):
            expr = sp.sympify(expr)

        free_syms = getattr(expr, "free_symbols", set())

        # Definir/asociar símbolos de trabajo
        s_match = [sym for sym in free_syms if sym.name == var_s]
        s_sym = s_match[0] if s_match else sp.symbols(var_s)

        t_match = [sym for sym in free_syms if sym.name == var_t]
        t_sym = t_match[0] if t_match else sp.symbols(var_t, real=True, positive=True)

        # 1. Detectar el dominio de la expresión ingresada y convertir al otro
        if any(sym.name == var_s for sym in free_syms) or s_sym in free_syms:
            X_s = sp.simplify(expr)
            try:
                x_t = sp.inverse_laplace_transform(X_s, s_sym, t_sym)
                x_t = x_t.replace(sp.Heaviside, lambda z: 1)
            except Exception:
                x_t = None
        else:
            x_t = expr
            try:
                res = sp.laplace_transform(expr, t_sym, s_sym)
                lap = res[0] if isinstance(res, tuple) else res
                X_s = sp.simplify(lap)
            except Exception:
                X_s = None

        # 2. Calcular las raíces (Polos y Ceros) desde X(s)
        polos, ceros = [], []
        if X_s is not None:
            num, den = sp.fraction(X_s)
            ceros_sym = sp.solve(num, s_sym)
            polos_sym = sp.solve(den, s_sym)

            if isinstance(ceros_sym, dict):
                ceros_sym = list(ceros_sym.values())
            if isinstance(polos_sym, dict):
                polos_sym = list(polos_sym.values())

            for c in ceros_sym:
                try:
                    ceros.append(complex(c.evalf()))
                except Exception:
                    pass

            for p in polos_sym:
                try:
                    polos.append(complex(p.evalf()))
                except Exception:
                    pass

        # 3. Generar la figura con 2 subplots lado a lado
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # --- Subplot 1: Dominio del Tiempo x(t) ---
        ax_t = axes[0]
        if x_t is not None:
            f_num = sp.lambdify(t_sym, x_t, modules=['numpy', 'sympy'])
            t_vals = np.linspace(t_range[0], t_range[1], num_pts)
            try:
                y_vals = f_num(t_vals)
                if np.isscalar(y_vals):
                    y_vals = np.full_like(t_vals, y_vals)
                ax_t.plot(t_vals, y_vals, color='#1f77b4', linewidth=2, label=r'$x(t)$')
            except Exception as err:
                ax_t.text(0.5, 0.5, f"Error al evaluar en tiempo:\n{err}", ha='center', va='center')
        else:
            ax_t.text(0.5, 0.5, "No se pudo calcular la Inversa de Laplace", ha='center', va='center')

        ax_t.set_title(r"Dominio del Tiempo: $x(t)$", fontsize=12, fontweight='bold')
        ax_t.set_xlabel(r"Tiempo ($t$)", fontsize=10)
        ax_t.set_ylabel(r"Amplitud", fontsize=10)
        ax_t.grid(True, linestyle=':', alpha=0.6)
        ax_t.axhline(0, color='black', linewidth=1, alpha=0.7)
        ax_t.legend(loc='upper right')

        # --- Subplot 2: Dominio de Laplace (Plano s) ---
        ax_s = axes[1]
        ax_s.axhline(0, color='black', linewidth=1.2)
        ax_s.axvline(0, color='black', linewidth=1.2)

        if show_unit_circle:
            theta = np.linspace(0, 2 * np.pi, 300)
            ax_s.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label=r'Círculo Unitario ($|s|=1$)')

        # Dibujar Polos ('X' rojas)
        if polos:
            ax_s.scatter([p.real for p in polos], [p.imag for p in polos],
                         color='red', marker='x', s=120, linewidth=2.5, zorder=5, label='Polos (X)')

        # Dibujar Ceros ('O' azules)
        if ceros:
            ax_s.scatter([c.real for c in ceros], [c.imag for c in ceros],
                         color='blue', marker='o', s=120, facecolors='none', linewidth=2, zorder=5, label='Ceros (O)')

        # Ajuste de escala automática del plano s
        puntos = polos + ceros
        if puntos:
            max_re = max([abs(p.real) for p in puntos] + [1.5])
            max_im = max([abs(p.imag) for p in puntos] + [1.5])
            lim = max(max_re, max_im) * 1.3
            ax_s.set_xlim(-lim, lim)
            ax_s.set_ylim(-lim, lim)

        ax_s.set_title(r"Dominio de Laplace: Plano $s$ (Polos y Ceros)", fontsize=12, fontweight='bold')
        ax_s.set_xlabel(r"Parte Real ($\sigma$)", fontsize=10)
        ax_s.set_ylabel(r"Parte Imaginaria ($j\omega$)", fontsize=10)
        ax_s.grid(True, linestyle=':', alpha=0.6)
        ax_s.set_aspect('equal', adjustable='box')
        ax_s.legend(loc='lower right')

        plt.tight_layout()
        plt.show()

        return X_s, x_t, polos, ceros

    # Alias equivalente a la función nativa de MATLAB
    pzmap = plot_pz