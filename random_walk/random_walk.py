#!/usr/bin/python

from random import choice


class RandomWalk:

    def __init__(self, num_points=5000):
        # initialize random walk properties
        self.num_points = num_points

        # random walk starts from (0, 0)
        self.x_values = [0]
        self.y_values = [0]

    def fill_walk(self):
        # calc all dots that random walk contains
        while len(self.x_values) < self.num_points:
            # decide direction and distance
            x_direction = choice([1, -1])
            x_distance = choice([0, 1, 2, 3, 4])
            x_step = x_direction * x_distance

            y_direction = choice([1, -1])
            y_distance = choice([0, 1, 2, 3, 4])
            y_step = y_direction * y_distance

            # prevent staying still
            if x_step == 0 and y_step == 0:
                continue

            # calc x and y value of next dot
            next_x = self.x_values[-1] + x_step
            next_y = self.y_values[-1] + y_step

            self.x_values.append(next_x)
            self.y_values.append(next_y)


'''
this is a comment

x_values = list(range(0, 5000))
y_values = [x**3 for x in x_values]

plt.scatter(x_values, y_values, edgecolors=(0.4, 0.1, 0.7), s=10)

# 设置标题与标签
plt.title("Square Numbers", fontsize=24)
plt.xlabel("Value", fontsize=14)
plt.ylabel("Square of Value", fontsize=14)

# 设置坐标轴刻度
plt.tick_params(axis="both", labelsize=12)

# 设置坐标轴的取值范围
plt.axis([0, 5000, 0, 125000000000])

# 显示
plt.show()
'''
