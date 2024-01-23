import numpy as np
import matplotlib.pyplot as plt

# Функция для нахождения коэффициентов кубического сплайна методом прогонки
# 1. Определение коэффициентов кубического сплайна методом прогонки:
#    - Функция cubic_spline_coefficients принимает массивы точек x и соответствующих им значений функции y,
#  использует метод прогонки для нахождения коэффициентов кубического сплайна и возвращает эти коэффициенты b, c, d.
def cubic_spline_coefficients(x, y):
    n = len(x)
    h = np.diff(x)
    alpha = np.zeros(n)
    for i in range(1, n - 1):
        alpha[i] = 3/h[i]*(y[i + 1] - y[i]) - 3/h[i - 1]*(y[i] - y[i - 1])

    l, mu, z = np.zeros(n), np.zeros(n), np.zeros(n)
    l[0] = 1
    mu[0] = 0
    z[0] = 0

    for i in range(1, n - 1):
        l[i] = 2*(x[i + 1] - x[i - 1]) - h[i - 1]*mu[i - 1]
        mu[i] = h[i]/l[i]
        z[i] = (alpha[i] - h[i - 1]*z[i - 1])/l[i]

    b, c, d = np.zeros(n), np.zeros(n), np.zeros(n)
    l[n - 1] = 1
    z[n - 1] = 0
    c[n - 1] = 0

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j]*c[j + 1]
        b[j] = (y[j + 1] - y[j])/h[j] - h[j]*(c[j + 1] + 2*c[j])/3
        d[j] = (c[j + 1] - c[j])/(3*h[j])

    return b, c, d

# Функция для вычисления значений кубического сплайна в точках интерполяции
# 2. Интерполяция кубическим сплайном:
#    - Функция cubic_spline_interpolation применяет полученные коэффициенты к точкам интерполяции x_interp 
# для вычисления значений интерполированной функции.
def cubic_spline_interpolation(x, y, x_interp):
    b, c, d = cubic_spline_coefficients(x, y)
    y_interp = []
    for x_int in x_interp:
        i = np.searchsorted(x, x_int) - 1
        if i == len(x) - 1:
            i -= 1
        dx = x_int - x[i]
        y_interp.append(y[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3)
    return np.array(y_interp)

# Создание исходных данных для интерполяции
x_data = np.array([1, 2, 3, 4, 5])  # Пример неравномерной сетки
y_data = np.array([0, 3, 2, 5, 1])  # Пример значений функции на этой сетке

# Генерация значений для интерполяции
x_interp = np.linspace(1, 5, 100)
y_interp = cubic_spline_interpolation(x_data, y_data, x_interp)

# Визуализация результатов
plt.figure(figsize=(10, 5))
plt.plot(x_data, y_data, 'ro', label='Original Data')
plt.plot(x_interp, y_interp, 'b-', label='Interpolated Data (Cubic Spline)')
plt.title('Cubic Spline Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

