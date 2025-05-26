import tkinter as tk
from tkinter import ttk
from sympy import symbols, Function
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.lines import Line2D  # 🔹 Для легенды

class ToolTip:
    def __init__(self, widget, text='tooltip'):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, background="#ffffe0", relief="solid", borderwidth=1, font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()

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

    x_vals = np.linspace(x_start, x_end, 400)
    C1_values = np.arange(c1_start, c1_end + c1_step, c1_step)

    fig, ax = plt.subplots(figsize=(8, 8))

    
    legend_elements = []


    for i, c in enumerate(C1_values):
        y_vals = c * x_vals**n_val
        l, = ax.plot(x_vals, y_vals, 'blue', alpha=0.7)
        if i == 0:
            legend_elements.append(Line2D([0], [0], color='blue', label='y = C₁·xⁿ'))

    
    xx, yy = np.meshgrid(x_vals, np.linspace(y_start, y_end, 400))
    for i, c in enumerate(C1_values):
        C_val = c
        zz = yy**2 + (1 / (n_val+0.0001)) * xx**2 - C_val
        ax.contour(xx, yy, zz, levels=[0], colors='red', alpha=0.6)
        if i == 0:
            legend_elements.append(Line2D([0], [0], color='red', label='Аналитические ортогональные'))

   
    def dy_dx_orthogonal(x_val, y_val):
        eps = 1e-6
        denom = n_val * y_val
        if abs(y_val) < eps or abs(denom) < eps:
            return 0.0
        return -x_val / denom

    numeric_color = {'Эйлера': 'orange', 'RK2': 'deeppink', 'RK4': 'lime'}
    numeric_label = {'Эйлера': 'ортогональные найденные численно(метод Эйлера)', 'RK2': 'ортогональные найденные численно(метод РК2)', 'RK4': 'ортогональные найденные численно(метод РК4)'}
    numeric_drawn = False

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
        else:
            rk_method = 'RK23' if method == 'RK2' else 'RK45'
            try:
                sol_f = solve_ivp(dy_dx_orthogonal, [x0, x_end], [y0], method=rk_method, max_step=0.1)
                sol_b = solve_ivp(dy_dx_orthogonal, [x0, x_start], [y0], method=rk_method, max_step=0.1)
                x_full = np.concatenate([sol_b.t[::-1], sol_f.t])
                y_full = np.concatenate([sol_b.y[0][::-1], sol_f.y[0]])
            except Exception:
                continue

        color = numeric_color[method]
        l, = ax.plot(x_full, y_full, color=color, alpha=0.7)
        if not numeric_drawn:
            legend_elements.append(Line2D([0], [0], color=color, label=numeric_label[method]))
            numeric_drawn = True

    ax.set_xlim([x_start, x_end])
    ax.set_ylim([y_start, y_end])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.show()

root = tk.Tk()
root.title("Ортогональные траектории")

def add_entry(label, row, default=""):
    tk.Label(root, text=label).grid(row=row, column=0, sticky="e")
    entry = tk.Entry(root)
    entry.insert(0, default)
    entry.grid(row=row, column=1)
    return entry

n_entry = add_entry("Степень n:", 0, "2")
ToolTip(n_entry, "Показатель степени n для функции y = C₁·xⁿ")

xmin_entry = add_entry("x min:", 1, "-5")
ToolTip(xmin_entry, "Минимальное значение по оси X для отображения графика")

xmax_entry = add_entry("x max:", 2, "5")
ToolTip(xmax_entry, "Максимальное значение по оси X для отображения графика")

ymin_entry = add_entry("y min:", 3, "-5")
ToolTip(ymin_entry, "Минимальное значение по оси Y для отображения графика")

ymax_entry = add_entry("y max:", 4, "5")
ToolTip(ymax_entry, "Максимальное значение по оси Y для отображения графика")


c1_min_entry = add_entry("C₁ min:", 5, "-10")
ToolTip(c1_min_entry, "Минимальное значение параметра C₁ в семействе кривых y = C₁·xⁿ")

c1_max_entry = add_entry("C₁ max:", 6, "10")
ToolTip(c1_max_entry, "Максимальное значение параметра C₁ в семействе кривых y = C₁·xⁿ")

c1_step_entry = add_entry("Шаг по C₁:", 7, "1")
ToolTip(c1_step_entry, "Шаг изменения параметра C₁ при построении семейства кривых")

step_entry = add_entry("Шаг интегрирования:", 8, "0.01")
ToolTip(step_entry, "Шаг по x для численного метода при построении ортогональных кривых")

tk.Label(root, text="Метод:").grid(row=9, column=0, sticky="e")
method_var = tk.StringVar(value="RK4")
method_menu = ttk.Combobox(root, textvariable=method_var, values=["Эйлера", "RK2", "RK4"], state="readonly")
method_menu.grid(row=9, column=1)
ToolTip(method_menu, "Выберите численный метод для построения ортогональных кривых")

tk.Button(root, text="Построить графики", command=run_combined_plot).grid(row=10, column=0, columnspan=2, pady=10)

root.mainloop()
