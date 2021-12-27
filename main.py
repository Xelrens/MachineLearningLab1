import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from networkx import all_pairs_shortest_path_length
from collections import defaultdict
import json
from pandas import  json_normalize, read_xml

# Часть 1
# Задание 1
# Преобразовать JSON-файл в CSV-файл.
# Решение:

def transformation():
    with open('sales.json') as str_json:
        input_json = json.load(str_json)

    transformed_json = []
    i = 0

    for item in input_json:
        item_name = item["item"]
        for country_sales in item["sales_by_country"]:
            for year_sales in item["sales_by_country"][country_sales]:
                transformed_json.append(dict())
                transformed_json[i]["item"] = item_name
                transformed_json[i]["country"] = country_sales
                transformed_json[i]["year"] = year_sales
                transformed_json[i]["sales"] = item["sales_by_country"][country_sales][year_sales]
                i += 1
    pure_json = json_normalize(transformed_json)
    pure_json.to_csv("sales.csv", index=False, sep=",")

transformation()

# Часть 2
# Задание 1
# Вычислить функцию.
# Решение:

count = 10

x = np.random.randint(low=-10, high=10, size=(count))
w = np.random.uniform(size=(count))
b = np.random.randint(low=-10, high=10)

print(np.dot(x, w) + b)

# Задание 2
# На отрезке [-5;5] построить графики функций.
# Решение:

plt.style.use("seaborn");

leftborder = -5
rightborder = 5

x = np.linspace(leftborder, rightborder, num=10) #разбивает интервал на 10 частей
f = (x - (x ** 3) / 6 + (x ** 5) / 120 - (x ** 7) / 5040)
g = np.sin(x)

plt.plot(x, f, label="f(x)", color="red")
plt.plot(x, g, label="g(x)", color="blue")
plt.xlim(leftborder, rightborder)
plt.legend(loc="upper center")
plt.show() 

# Задание 3
# Нарисовать в matplotlib окружность заданного радиуса r.
# Решение:

figure, axes = plt.subplots()
r = float(input()) #Радиус
x = np.linspace(-r, r, 100)
f = lambda x: -(r ** 2 - x ** 2) ** 0.5
g = lambda x: (r ** 2 - x ** 2) ** 0.5

plt.plot(x, f(x), "red")
plt.plot(x,g(x), "red")
axes.set_aspect(1) 
plt.show()

# Задание 4
# Написать функцию transformation_plot, принимающую на вход набор двумерных точек и квадратную матрицу
# размером 2x2. Отрисуйте на одном графике оригинальные точки, на втором - точки после преобразования
# при помощи матрицы. Отобразите точки таким образом, чтобы было понятно, какая из точек на первом графике
# соответствует какой точке на втором.
# Решение:

def transformation_plot(points, matrix):
    x = points[:, 0]
    y = points[:, 1]

    color = x * y
 
    axes[0].scatter(x, y, 100, c=color) #больший масштаб был выбран для того, чтоб удобнее было сравнить точки 

    transformed_points = np.dot(points, matrix.T)
    transformed_x = transformed_points[:, 0] 
    transformed_y = transformed_points[:, 1]

    axes[1].scatter(transformed_x, transformed_y, 100, c=color)

figure, axes = plt.subplots(1, 2, figsize=(10, 10))

transformation_plot(np.random.random(size=(200, 2)), np.array([[4, 3], [5, 2]]))
plt.show()

# Задание 5
# Задайте некоторую функцию одной переменной f(x) (пример: sin(x), ln(x), x**2 + 2x + 1, …):
# Отрисуйте график её производной на выбранном интервале [a,b], не используя её аналитическое выражение.
# Сравните для проверки с аналитическим выражением производной.
# Решение:

def f(x):
    return np.sin(x)

def derivative_f(x):
    delta_x = 0.0001 #для вычисления производной
    return (f(x + delta_x) - f(x)) / delta_x

def analytic_derivative(x):
    return np.cos(x)

x = np.linspace(-10, 10, 10) #малое количество разбиений взято для большей наглядности графика

figure, axes = plt.subplots(1, 3, sharex = True, sharey = True, figsize = (15, 5))
axes[0].plot(x, f(x), label = "function_f", color = "red")
axes[1].plot(x, derivative_f(x), label = "derivative", color = "blue")
axes[2].plot(x, analytic_derivative(x), label = "analytic_derivative", color = "green")

axes[0].legend(loc = "upper right")
axes[1].legend(loc = "upper right")
axes[2].legend(loc = "upper right")

plt.show()

# Задание 6
# Дано множество векторов V размерности d. Дан вектор q такой же размерности. Определить:
# а) Пропорцию векторов v в V, для которых угол(v,q) < 90 градусов
# б) Пропорцию векторов v в V, для которых угол(v,q) < 30 градусов
# Решение:

d = int(input("Размерность векторов: "))
q = [1] * d
np.random.seed(0)

V = [[0] * d] * 100
for i in range(100):
    for j in range(d):
        V[i] = (np.random.random(d) - 0.5)

i = 0
j = 0
for v in V:
    angle = np.arccos(np.dot(q, v) /(np.linalg.norm(q) * np.linalg.norm(v)))
    if angle * 180/np.pi < 90:
        i += 1
    if angle * 180/np.pi < 30:
        j += 1

print("угол(v, q) < 90 градусов =", (i/100))
print("угол(v, q) < 30 градусов =", (j/100))

# Задание 7
# Дан гиперкуб и вписанная в него гиперсфера. Через сэмплинг точек внутри гиперкуба,
# оценить отношение объёма гиперсферы к объёму гиперкуба.
# Вывести график этой пропорции в зависимости от размерности пространства d.
# Решение:

array = []

for i in range(2, 10):
  V = [[0] * i] * 100
  for j in range(100):
      for k in range(i):
          V[j] = (np.random.random(i) - 0.5)
  incirc = 0
  for v in V:
      if np.linalg.norm(v) <= 0.5:
          incirc += 1
  array.append(incirc/100)
  i += 1

fig,ax = plt.subplots()
ax.set_xlim(2, 10)
ax.set_ylim(0, 1)
ax.plot(range(2, 10), array, 1)

fig.show()
plt.show()




