from __future__ import (absolute_import, division, print_function, unicode_literals)
from urllib.request import urlopen
import json
import pygal

json_url = 'https://raw.githubusercontent.com/muxuezi/btc/master/btc_close_2017.json'
response = urlopen(json_url)
# read data
req = response.read()
# write data into files
with open('btc_close_2017_urllib.json', 'wb') as f:
    f.write(req)
# load json format
file_urllib = json.loads(req)

# 实现每个元素换行输出
# for i in range(0, len(file_urllib)):
#     print(file_urllib[i])

# load data into a list
filename = 'btc_close_2017.json'
with open(filename) as f:
    btc_data = json.load(f)

# print daily info
for btc_dict in btc_data:
    date = btc_dict['date']
    month = int(btc_dict['month'])
    week = int(btc_dict['week'])
    weekday = btc_dict['weekday']
    close = int(float(btc_dict['close']))
    print("{} is month {}, week {}, {}, the close price is CNY {} ".format(date, month, week, weekday, close))

# create 5 lists to store data and close price
dates = []
months = []
weeks = []
weekdays = []
close = []

# everyday info
for btc_dict in btc_data:
    dates.append(btc_dict['date'])
    months.append(int(btc_dict['month']))
    weeks.append(int(btc_dict['week']))
    weekdays.append(btc_dict['weekday'])
    close.append(int(float(btc_dict['close'])))

# print chart
line_chart = pygal.Line(x_label_rotation=20, show_minor_x_labels=False)
line_chart.title = '收盘价（￥）'
line_chart.x_labels = dates
N = 20  # x轴坐标每20天显示一次
line_chart.x_labels_major = dates[::N]
line_chart.add("收盘价", close)
line_chart.render_to_file('收盘价折线图（￥）.svg')
