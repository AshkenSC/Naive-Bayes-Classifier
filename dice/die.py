from random import randint


class Die:
    # a class standing for dice

    def __init__(self, num_sides=6):
        # default: 6-side dice
        self.num_sides = num_sides

    def roll(self):
        # return a random value between 1 and dice sides
        return randint(1, self.num_sides)