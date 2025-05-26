import tkinter as tk
from tkinter import ttk
from sympy import symbols, Eq, Function, diff, solveset, dsolve, logcombine
from sympy.plotting import plot_implicit
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import tempfile
import matplotlib.image as mpimg  

def run_combined_plot():
    try:
        n_val = float(n_entry.get())
        x_start = float(xmin_entry.get())
        x_end = float(xmax_entry.get())
        y_start = float(ymin_entry.get())
        y_end = float(ymax_entry.get())
        c1_start = float(c1_min_entry.get())
        c1_end = float(c1_max_entry.get())
        c1_step = float(c1_step_entry.get())
        step = float(step_entry.get())
        method = method_var.get()
    except ValueError:
        print("Ошибка: заполните все поля числами.")
        return

    x, C1, n = symbols("x C1 n")
    y = Function("y")(x)
    Y = symbols("y")

    fam = Eq(y, C1 * x ** n)
    hyp = fam.subs(n, n_val)

    plots = []
    for c in np.arange(c1_start, c1_end, c1_step):
        f = hyp.subs({C1: c, y: Y})
        p = plot_implicit(f, (x, x_start, x_end), (Y, y_start, y_end), show=False, line_color='blue')
        plots.append(p)

    csol = solveset(fam, C1).args[0]
    ode = Eq(diff(csol, x), 0)
    dsol = dsolve(ode, y, hint='separable')
    dsol = logcombine(dsol, force=True)
    ode2 = ode.subs(diff(y, x), -1 / diff(y, x))
    sol = solveset(ode2, diff(y, x))
    rhs2 = sol.args[0].args[0]
    ode2 = Eq(diff(y, x), rhs2)
    dsol2 = dsolve(ode2, y, hint="separable", simplify=False)

    for c in np.arange(c1_start, c1_end, c1_step):
        f = dsol2.subs({C1: c, y: Y, n: n_val})
        p = plot_implicit(f, (x, x_start, x_end), (Y, y_start, y_end), show=False, line_color='red')
        plots.append(p)

    if plots:
        base_plot = plots[0]
        for p in plots[1:]:
            base_plot.extend(p)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            png_path = tmpfile.name
        base_plot.save(png_path)


        img = mpimg.imread(png_path)

    def dy_dx_orthogonal(x_val, y_val):
        eps = 1e-6
        denom = n_val * y_val
        if abs(y_val) < eps or abs(denom) < eps:
            return 0.0
        return -x_val / denom

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))


    if plots:
        axs[0].imshow(img, extent=[x_start, x_end, y_start, y_end], aspect='auto', origin='upper')
        
        axs[0].axis('tight')
        axs[0].set_title("Аналитическое решение")
        axs[0].set_xlim([x_start, x_end])
        axs[0].set_ylim([y_start, y_end])
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        axs[0].grid(False)


    C1_values = np.arange(c1_start, c1_end, c1_step)
    x_vals = np.linspace(x_start, x_end, 300)

    for C1v in C1_values:
        with np.errstate(invalid='ignore', divide='ignore'):
            y_vals = C1v * x_vals ** n_val
            axs[1].plot(x_vals, y_vals, 'b', alpha=0.3)

    for C1v in C1_values:
        x0 = 2.0 if n_val < 0 else 1.0
        y0 = C1v * x0 ** n_val

        if abs(y0) < 1e-6 or not np.isfinite(y0):
            continue

        if method == 'Эйлера':
            def euler_method(f, x0, y0, h, steps):
                x_vals = [x0]
                y_vals = [y0]
                for _ in range(steps):
                    dy = f(x_vals[-1], y_vals[-1])
                    y_new = y_vals[-1] + h * dy
                    x_new = x_vals[-1] + h
                    if not np.isfinite(y_new) or abs(y_new) > 1e6:
                        break
                    x_vals.append(x_new)
                    y_vals.append(y_new)
                return np.array(x_vals), np.array(y_vals)

            xf, yf = euler_method(dy_dx_orthogonal, x0, y0, step, int((x_end - x0) / step))
            xb, yb = euler_method(dy_dx_orthogonal, x0, y0, -step, int((x0 - x_start) / step))
            x_full = np.concatenate([xb[::-1], xf])
            y_full = np.concatenate([yb[::-1], yf])
            color = 'red'
        else:
            rk_method = 'RK23' if method == 'RK2' else 'RK45'
            try:
                sol_f = solve_ivp(dy_dx_orthogonal, [x0, x_end], [y0], method=rk_method, max_step=0.1)
                sol_b = solve_ivp(dy_dx_orthogonal, [x0, x_start], [y0], method=rk_method, max_step=0.1)
                x_full = np.concatenate([sol_b.t[::-1], sol_f.t])
                y_full = np.concatenate([sol_b.y[0][::-1], sol_f.y[0]])
            except Exception:
                continue
            color = 'deeppink' if method == 'RK2' else 'lime'

        axs[1].plot(x_full, y_full, color=color, alpha=0.7)

    axs[1].set_title(f"Численное решение ({method})")
    axs[1].set_xlim([x_start, x_end])
    axs[1].set_ylim([y_start, y_end])
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].set_aspect('auto')
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()

    if os.path.exists(png_path):
        os.remove(png_path)


root = tk.Tk()
root.title("Ортогональные траектории")

def add_entry(label, row, default=""):
    tk.Label(root, text=label).grid(row=row, column=0, sticky="e")
    entry = tk.Entry(root)
    entry.insert(0, default)
    entry.grid(row=row, column=1)
    return entry

n_entry = add_entry("Степень n:", 0, "2")
xmin_entry = add_entry("x min:", 1, "-5")
xmax_entry = add_entry("x max:", 2, "5")
ymin_entry = add_entry("y min:", 3, "-5")
ymax_entry = add_entry("y max:", 4, "5")
c1_min_entry = add_entry("C₁ min:", 5, "-10")
c1_max_entry = add_entry("C₁ max:", 6, "10")
c1_step_entry = add_entry("Шаг по C₁:", 7, "1")
step_entry = add_entry("Шаг интегрирования:", 8, "0.01")

tk.Label(root, text="Метод:").grid(row=9, column=0, sticky="e")
method_var = tk.StringVar(value="RK4")
method_menu = ttk.Combobox(root, textvariable=method_var, values=["Эйлера", "RK2", "RK4"], state="readonly")
method_menu.grid(row=9, column=1)

tk.Button(root, text="Построить графики", command=run_combined_plot).grid(row=10, column=0, columnspan=2, pady=10)

root.mainloop()
