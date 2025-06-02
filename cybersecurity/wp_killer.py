from io import BytesIO
from lxml import etree
from queue import Queue

import requests
import sys
import threading
import time

SUCCESS = "Welcome to WordPress!"
TARGET = "http://127.0.0.1:31337/wp-login.php"
WORDLIST = 'cain.txt'

def get_words():
    with open(WORDLIST) as f:
        raw_words = f.read()

    words = Queue()
    for word in raw_words.split():
        words.put(word)

    return words

class Bruter:
    def __init__(self, username, url):
        self.username = username
        self.url = url
        self.found = False
        print(f"URL is {url}\n")
        print("username = %s\n" % username )

    def run_bruteforce(self, passwords):
        print("------run bruteforce-------")
        print(passwords.get())
            

def run():
    print("---------run----------")
    b = Bruter("admin", TARGET)
    words = get_words()
    b.run_bruteforce(words)

if __name__ == "__main__":
    run()
