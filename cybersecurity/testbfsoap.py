from bs4 import BeautifulSoup as bs
import requests

#url = 'http://bing.com'

def run():
    url = 'https://www.google.com'

    r = requests.get(url)
    tree = bs(r.text, 'html.parser')

    for link in tree.find_all('a'):
        print(f"{link.get('href')} -> {link.text}")

if __name__ == "__main__":
    print("--------Start---------")
    run()
    print("--------End---------")    
    
