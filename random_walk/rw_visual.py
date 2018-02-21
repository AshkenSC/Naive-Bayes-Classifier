import matplotlib.pyplot as plt

from random_walk import RandomWalk

# create a RandomWalk instance
while True:
    rw = RandomWalk(5000)
    rw.fill_walk()

    # set dots' attributes
    point_numbers = list(range(rw.num_points))
    # plt.scatter(rw.x_values, rw.y_values, c=point_numbers, cmap=plt.cm.Blues, edgecolor='none', s=1)
    plt.plot(rw.x_values, rw.y_values, linewidth=1)

    # hide axes
    # plt.axes().get_xaxis().set_visible(False);
    # plt.axes().get_yaxis().set_visible(False);

    plt.show()

    keep_running = input("Another walk? Y/N")
    if keep_running == 'n' or 'N':
        break
