import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
import random

from matplotlib import colormaps
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sympy import (
    symbols, sympify, lambdify, diff, solve, solveset, Eq, S, N,
    im, oo, sstr
)
from sympy.calculus.util import continuous_domain, function_range
from sympy.calculus.singularities import singularities
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

# разрешаем "2x", "3sin(x)", "x^2" и т.п. вместо строгого "2*x", "3*sin(x)", "x**2"
_PARSE_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application, convert_xor,
)

# ---------------------------------------------------------------------------
# базовая настройка
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

cmap = colormaps["tab20"]

X = symbols("x", real=True)

# каждый элемент: {"str": исходная строка, "expr": sympy-выражение,
#                   "f": numpy-функция, "color": цвет, "analysis": dict}
functions = []


def parse_func_str(func_str):
    """Разбирает строку в sympy-выражение, разрешая неявное умножение (2x, x^2 и т.п.)."""
    return parse_expr(func_str, local_dict={"x": X}, transformations=_PARSE_TRANSFORMATIONS)

window = ctk.CTk()                              # окно для поля ввода функции
window.geometry("260x130+180+110")
window.title("Введите функцию")

BASE_WIDTH = 800
BASE_HEIGHT = 700
BASE_X = 10
BASE_Y = 10


# ---------------------------------------------------------------------------
# вспомогательные функции для математического анализа
# ---------------------------------------------------------------------------
def is_real_number(val):
    """Проверяет, что sympy-значение можно трактовать как вещественное число."""
    try:
        c = complex(N(val))
        return abs(c.imag) < 1e-9
    except Exception:
        return False


def fmt_num(val):
    """Красиво форматирует число: точная форма + десятичное приближение."""
    exact = sstr(val)
    try:
        approx = float(N(val))
        if abs(approx - round(approx)) < 1e-9:
            approx_str = str(round(approx))
        else:
            approx_str = f"{approx:.4f}"
    except Exception:
        approx_str = "?"
    if exact == approx_str:
        return exact
    return f"{exact} ≈ {approx_str}"


def fmt_interval(iv):
    left = "(" if iv.left_open else "["
    right = ")" if iv.right_open else "]"
    a = "-∞" if iv.start == -oo else fmt_num(iv.start)
    b = "∞" if iv.end == oo else fmt_num(iv.end)
    return f"{left}{a}; {b}{right}"


def fmt_set(s):
    """Приводит sympy-множество (интервал/объединение) к читаемой строке вида (a; b) ∪ (c; d)."""
    if s is None:
        return "не удалось вычислить"
    try:
        from sympy import Interval, Union, EmptySet, FiniteSet
        if s == S.Reals:
            return "все действительные числа (ℝ)"
        if s == EmptySet:
            return "пусто (∅)"
        if isinstance(s, Interval):
            return fmt_interval(s)
        if isinstance(s, Union):
            parts = []
            for arg in s.args:
                if isinstance(arg, Interval):
                    parts.append(fmt_interval(arg))
                else:
                    parts.append(str(arg))
            return " ∪ ".join(parts)
        if isinstance(s, FiniteSet):
            return "{" + ", ".join(fmt_num(v) for v in s.args) + "}"
        return str(s).replace("oo", "∞")
    except Exception:
        return str(s).replace("oo", "∞")


def analyze_function(func_str):
    """Полный математический анализ функции по её строковому представлению."""
    result = {}
    try:
        expr = parse_func_str(func_str)
    except Exception as e:
        return {"error": f"Не удалось распознать функцию: {e}"}

    result["expr"] = expr

    # --- область определения ---------------------------------------------
    try:
        result["domain"] = continuous_domain(expr, X, S.Reals)
    except Exception:
        result["domain"] = None

    # --- область значений ---------------------------------------------
    try:
        result["range"] = function_range(expr, X, S.Reals)
    except Exception:
        result["range"] = None

    # --- точки пересечения с осями -----------------------------------------
    try:
        raw_roots = solve(Eq(expr, 0), X)
        result["x_intercepts"] = [r for r in raw_roots if is_real_number(r)]
    except Exception:
        result["x_intercepts"] = []

    try:
        y0 = expr.subs(X, 0)
        result["y_intercept"] = y0 if is_real_number(y0) else None
    except Exception:
        result["y_intercept"] = None

    # --- точки разрыва -------------------------------------------------------
    try:
        result["discontinuities"] = sorted(
            [p for p in singularities(expr, X) if is_real_number(p)],
            key=lambda v: float(N(v)),
        )
    except Exception:
        result["discontinuities"] = []

    # --- производная -----------------------------------------------------
    try:
        derivative = diff(expr, X)
        result["derivative"] = derivative
    except Exception:
        derivative = None
        result["derivative"] = None

    # --- критические точки -------------------------------------------------
    critical_points = []
    if derivative is not None:
        try:
            raw_cp = solve(Eq(derivative, 0), X)
            critical_points = sorted(
                [c for c in raw_cp if is_real_number(c)],
                key=lambda v: float(N(v)),
            )
        except Exception:
            critical_points = []
    result["critical_points"] = critical_points

    # --- минимумы / максимумы (через 2-ю производную) ----------------------
    extrema = []
    if derivative is not None and critical_points:
        second = diff(derivative, X)
        for cp in critical_points:
            try:
                y_val = expr.subs(X, cp)
                if not is_real_number(y_val):
                    continue
                sd_val = second.subs(X, cp)
                if is_real_number(sd_val):
                    sd_f = float(N(sd_val))
                    if sd_f > 1e-9:
                        kind = "минимум"
                    elif sd_f < -1e-9:
                        kind = "максимум"
                    else:
                        kind = "точка перегиба / требует доп. анализа"
                else:
                    kind = "требует доп. анализа"
                extrema.append((cp, y_val, kind))
            except Exception:
                pass
    result["extrema"] = extrema

    # --- промежутки знакопостоянства -----------------------------------------
    try:
        result["positive_intervals"] = solveset(expr > 0, X, domain=S.Reals)
    except Exception:
        result["positive_intervals"] = None
    try:
        result["negative_intervals"] = solveset(expr < 0, X, domain=S.Reals)
    except Exception:
        result["negative_intervals"] = None

    return result


def format_analysis(func_str, a):
    """Готовит текстовый блок анализа для вывода в панель информации."""
    lines = [f"── f(x) = {func_str} ──"]

    if "error" in a:
        lines.append(a["error"])
        return "\n".join(lines) + "\n"

    lines.append(f"Производная: f'(x) = {sstr(a['derivative']) if a['derivative'] is not None else '—'}")

    lines.append("")
    lines.append(f"Область определения: {fmt_set(a['domain']) if a['domain'] is not None else 'не удалось вычислить'}")
    lines.append(f"Область значений:    {fmt_set(a['range']) if a['range'] is not None else 'не удалось вычислить'}")

    lines.append("")
    if a["x_intercepts"]:
        roots = ", ".join(fmt_num(r) for r in a["x_intercepts"])
        lines.append(f"Точки пересечения с Ox: x = {roots}")
    else:
        lines.append("Точки пересечения с Ox: не найдены (или не удалось решить аналитически)")
    if a["y_intercept"] is not None:
        lines.append(f"Точка пересечения с Oy: y = {fmt_num(a['y_intercept'])}")
    else:
        lines.append("Точка пересечения с Oy: нет (или x=0 вне области определения)")

    lines.append("")
    if a["discontinuities"]:
        pts = ", ".join(fmt_num(p) for p in a["discontinuities"])
        lines.append(f"Точки разрыва: x = {pts}")
    else:
        lines.append("Точки разрыва: не найдены")

    lines.append("")
    if a["critical_points"]:
        pts = ", ".join(fmt_num(c) for c in a["critical_points"])
        lines.append(f"Критические точки: x = {pts}")
    else:
        lines.append("Критические точки: не найдены")

    if a["extrema"]:
        lines.append("Вершины / экстремумы:")
        for cp, yv, kind in a["extrema"]:
            lines.append(f"   x = {fmt_num(cp)},  y = {fmt_num(yv)}  →  {kind}")
    else:
        lines.append("Вершины / экстремумы: не найдены")

    lines.append("")
    pos = fmt_set(a["positive_intervals"]) if a["positive_intervals"] is not None else "не удалось вычислить"
    neg = fmt_set(a["negative_intervals"]) if a["negative_intervals"] is not None else "не удалось вычислить"
    lines.append(f"f(x) > 0 при: {pos}")
    lines.append(f"f(x) < 0 при: {neg}")

    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# логика добавления/отрисовки функций
# ---------------------------------------------------------------------------
def add_function():                 # добавление функции
    func_str = e.get().strip()
    if not func_str:
        return

    try:
        expr = parse_func_str(func_str)
        f_numeric = lambdify(X, expr, modules=["numpy"])
    except Exception as error:
        info.deiconify()
        textbox.configure(state="normal")
        textbox.insert("end", f"── f(x) = {func_str} ──\nОшибка разбора: {error}\n\n")
        textbox.configure(state="disabled")
        return

    color = random.choice(cmap.colors)
    analysis = analyze_function(func_str)

    functions.append({
        "str": func_str,
        "expr": expr,
        "f": f_numeric,
        "color": color,
        "analysis": analysis,
    })

    app.deiconify()
    info.deiconify()

    textbox.configure(state="normal")
    textbox.insert("end", format_analysis(func_str, analysis))
    textbox.configure(state="disabled")
    textbox.see("end")

    e.delete(0, "end")
    update_graph()


def update_graph():                 # обновление графика
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if xlim == (0.0, 1.0) and ylim == (0.0, 1.0):
        xlim = (-4.0, 4.0)
        ylim = (-4.0, 4.0)
    ax.clear()
    xmin, xmax = xlim
    x_vals = np.linspace(xmin, xmax, 2000)

    for item in functions:
        try:
            with np.errstate(all="ignore"):
                y_vals = item["f"](x_vals)
                y_vals = np.asarray(y_vals, dtype=float)
                if y_vals.shape == ():
                    y_vals = np.full_like(x_vals, y_vals)
            ax.plot(x_vals, y_vals, color=item["color"], zorder=10, label=item["str"])
        except Exception as error:
            print("Ошибка построения:", error)

        # отмечаем вершины/экстремумы точками на графике
        for cp, yv, kind in item["analysis"].get("extrema", []):
            try:
                cp_f = float(N(cp))
                yv_f = float(N(yv))
                if xmin <= cp_f <= xmax:
                    ax.plot(cp_f, yv_f, "o", color=item["color"], zorder=11, markersize=6)
            except Exception:
                pass

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.grid(True)
    ax.minorticks_on()

    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    ax.spines["bottom"].set_zorder(1)
    ax.spines["left"].set_zorder(1)

    ax.spines["bottom"].set_color("#C5C5C5")
    ax.spines["left"].set_color("#C5C5C5")

    # фон
    ax.set_facecolor("#191919")
    # торец
    figure.set_facecolor("#191919")
    # большая сетка
    ax.grid(which="major", color="#a9a9a9", linewidth=0.8, zorder=0)
    # мелкая сетка
    ax.grid(which="minor", color="#393939", linewidth=0.4, zorder=0)
    # цифры на осях
    ax.tick_params(axis="both", colors="white")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    canvas.draw()


def clear_graph():                  # удаление всех функций
    functions.clear()
    ax.clear()
    canvas.draw()
    textbox.configure(state="normal")
    textbox.delete("1.0", "end")
    textbox.configure(state="disabled")
    app.withdraw()
    info.withdraw()


last_size = [0, 0]
PIXELS_PER_UNIT = 80  # пикселей на одну единицу оси


def resize_graph(event):
    if event.widget != app:
        return

    width = event.width
    height = event.height

    if [width, height] == last_size:
        return
    last_size[0] = width
    last_size[1] = height

    half_x = width / 2 / PIXELS_PER_UNIT
    half_y = height / 2 / PIXELS_PER_UNIT

    ax.set_xlim(-half_x, half_x)
    ax.set_ylim(-half_y, half_y)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))

    ax.minorticks_on()
    update_graph()


# ---------------------------------------------------------------------------
# окно графика
# ---------------------------------------------------------------------------
app = ctk.CTkToplevel(window)
app.withdraw()
app.geometry("800x800+861+110")
app.minsize(800, 800)
app.title("График функции")
app.bind("<Configure>", resize_graph)
# закрытие крестиком просто прячет окно, а не уничтожает его —
# иначе последующие обращения к app/canvas/ax падают с TclError
app.protocol("WM_DELETE_WINDOW", app.withdraw)

figure = plt.Figure(dpi=100)
ax = figure.add_subplot(111)
figure.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

canvas = FigureCanvasTkAgg(figure, master=app)
canvas.get_tk_widget().pack(fill="both", expand=True)


def zoom(event):
    factor = 0.9 if event.delta > 0 else 1.1  # колесо вверх — приближение, вниз — отдаление

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    cx = (xlim[0] + xlim[1]) / 2
    cy = (ylim[0] + ylim[1]) / 2

    new_half_x = (xlim[1] - xlim[0]) / 2 * factor
    new_half_y = (ylim[1] - ylim[0]) / 2 * factor

    ax.set_xlim(cx - new_half_x, cx + new_half_x)
    ax.set_ylim(cy - new_half_y, cy + new_half_y)

    update_graph()


canvas.get_tk_widget().bind("<MouseWheel>", zoom)

# ---------------------------------------------------------------------------
# окно ввода функции
# ---------------------------------------------------------------------------
e = ctk.CTkEntry(
    window,
    width=220,
    placeholder_text="Например: x**2 или sin(x)"
)
e.pack(pady=10)
e.bind("<Return>", lambda event: add_function())

b = ctk.CTkButton(window, text="Построить график", command=add_function)
b.pack(pady=(0, 5))

b_clear = ctk.CTkButton(window, text="Очистить всё", command=clear_graph, fg_color="#8B3A3A", hover_color="#6E2E2E")
b_clear.pack()

# ---------------------------------------------------------------------------
# окно для информации о функции (анализ)
# ---------------------------------------------------------------------------
info = ctk.CTkToplevel(window)
info.geometry("420x800+440+110")
info.withdraw()
info.title("Анализ функции")
# по той же причине, что и для app: крестик прячет, а не уничтожает окно
info.protocol("WM_DELETE_WINDOW", info.withdraw)

textbox = ctk.CTkTextbox(info, width=400, height=780, wrap="word", font=("Menlo", 13))
textbox.pack(padx=10, pady=10, fill="both", expand=True)
textbox.configure(state="disabled")

window.mainloop()