import csv
from matplotlib import pyplot as plt
from matplotlib import ticker
from datetime import datetime

# retrieve highest temperature from file
# and store data in highs[] using high as temp var
filename = 'sitka_weather_2014.csv'
with open(filename) as f:
    reader = csv.reader(f)
    header_row = next(reader)

    dates, highs, lows = [], [], []
    for row in reader:
        # store date in list dates[]
        current_date = datetime.strptime(row[0], "%Y-%m-%d")
        dates.append(current_date)
        # store highest temperature data in list highs[]
        high = int(row[1])
        highs.append(high)
        # store lowest temperature data in list lows[]
        low = int(row[3])
        lows.append(low)

    # print(highs)

# draw figure according to data

fig = plt.figure(dpi=128, figsize=(10, 6))
plt.plot(dates, highs, c='red')     # draw high line
plt.plot(dates, lows, c='blue')     # draw low line
plt.fill_between(dates, highs, lows, facecolor='blue', alpha=0.1)

# set figure format
plt.title("Daily high and low temperatures, 2014", fontsize=24)
plt.xlabel('', fontsize=16)
fig.autofmt_xdate();
plt.ylabel("Temperature(F)", fontsize=16)
plt.tick_params(axis='both', which='both', labelsize=16)

# set x-axis density
# tick_spacing = 2
# plt.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

plt.show()

    # for index, column_header in enumerate(header_row):
    #     print(index, column_header)
