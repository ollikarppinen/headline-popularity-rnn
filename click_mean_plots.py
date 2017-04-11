import util
import pandas as pd
import matplotlib.pyplot as plt

df = util.fetch.with_weekday_and_hour()

mean_by_weekday = df.groupby(['weekday']).mean().reset_index()
mean_by_weekday.to_csv('./clicks_by_weekday.csv')
mean_by_hour = df.groupby(['hour']).mean().reset_index()
mean_by_hour.to_csv('./clicks_by_hour.csv')

fig, ax = plt.subplots()
plt.bar(mean_by_weekday['weekday'], mean_by_weekday['clicks'])
plt.title('Vierailut viikonpäivän mukaan')
plt.ylabel('keskimääräiset vierailut')
plt.xlabel('viikonpäivä')
ax.set_xticklabels(('', 'ma', 'ti', 'ke', 'to', 'pe', 'la', 'su'))
plt.savefig('./clicks_by_weekday.pdf')
plt.clf()
plt.bar(mean_by_hour['hour'], mean_by_hour['clicks'])
plt.title('Vierailut tunnin mukaan')
plt.xlabel('tunti')
plt.ylabel('keskimääräiset vierailut')
plt.savefig('./clicks_by_hour.pdf')
