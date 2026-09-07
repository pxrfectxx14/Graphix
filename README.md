# Graphix

**[Русский](README.ru.md)** | **[English](README.md)**

**Graphix is a simple desktop app for plotting mathematical functions and analyzing their properties. Type a function — get a graph and a full breakdown of its properties.**
---

### Features

- Plot one or multiple functions at once
- Zoom and pan with the mouse wheel
- Domain and range
- X- and Y-intercepts
- Discontinuities
- Derivative and critical points
- Minima and maxima (marked on the graph)
- Sign intervals
- Implicit multiplication support (`2x`, `3sin(x)`, `x^2`)
---

### Requirements

- Python 3.9+
- `customtkinter`, `matplotlib`, `numpy`, `sympy`
---

### Installation

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install customtkinter matplotlib numpy sympy
```
---

### Usage

```bash
python main.py
```

Type a function (e.g. `x**2 - 4` or `sin(x)`) and click "Построить график" (Plot).
---