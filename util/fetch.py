#encoding: utf-8

import pandas as pd

def csv():
    csv = pd.read_csv("http://207.154.192.240/ampparit/ampparit-final.csv")
    return csv

def with_weekday_and_hour():
    df = csv()
    df['date_with_year'] = df['date'] + '2017.'
    df['date_object'] = pd.to_datetime(df['date_with_year'], format = '%H:%M, %d.%m.%Y.')
    df['weekday'] = df['date_object'].apply(lambda x: x.weekday())
    df['hour'] = df['date_object'].apply(lambda x: x.hour)
    return df
