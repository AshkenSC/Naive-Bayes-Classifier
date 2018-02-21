from dice.die import Die
import pygal

# create two dices
die_1 = Die()
die_2 = Die()

# throw dices and store results in a list
results = []
for roll_num in range(1000):
    result = die_1.roll()+die_2.roll()
    results.append(result)

# analyze results
frequencies = []
max_result = die_1.num_sides+die_2.num_sides
for value in range(1, max_result + 1):
    frequency = results.count(value)
    frequencies.append(frequency)

# visualize results
hist = pygal.Bar()

hist.title = "Results of rolling two D6 1000 times"
hist.x_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
hist.x_title = "Result"
hist.y_title = "Frequency of Result"

hist.add('D6 + D6', frequencies)
hist.render_to_file('die_visual.svg')

# print results
# for i in range(100):
#     print(results[i], end=' ')
#     if i % 10 == 0:
#         print("\n")
