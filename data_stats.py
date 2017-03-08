import util
import pandas as pd
import matplotlib.pyplot as plt

df = util.fetch.with_weekday_and_hour()

mean_by_weekday = df.groupby(['weekday']).mean().reset_index()
mean_by_weekday.to_csv('./clicks_by_weekday.csv')
mean_by_hour = df.groupby(['hour']).mean().reset_index()
mean_by_hour.to_csv('./clicks_by_hour.csv')

plt.bar(mean_by_weekday['weekday'], mean_by_weekday['clicks'])
plt.title('Clicks by weekday')
plt.ylabel('weekday, 0 == monday')
plt.xlabel('clicks mean')
plt.savefig('./clicks_by_weekday.pdf')
plt.clf()
plt.bar(mean_by_hour['hour'], mean_by_hour['clicks'])
plt.title('Clicks by hour')
plt.ylabel('hour')
plt.xlabel('clicks mean')
plt.savefig('./clicks_by_hour.pdf')
