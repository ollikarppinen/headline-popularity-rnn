#encoding: utf-8

def fetch_csv():
    csv = pandas.read_csv("http://207.154.192.240/ampparit/ampparit.csv")
    csv['title'] = csv['title'].str.lower()
    return csv
