import queue
import requests
import sys
import threading

AGENT="Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0"
EXTENSIONS=['.php','.bak','.orig','.inc']
TARGET="http://testphp.vulnweb.com"
THREADS=10
WORDLIST="/home/xxx/download/svndig/all.txt"

def get_words(resume=None):
    def extend_words(word):
        if "." in word:
            words.put(f'/{word}')
        else:
            words.put(f'/{word}/')

        for extension in EXTENSIONS:
            words.put(f'{word}{extension}')

    
    with open(WORDLIST) as f:
        raw_words = f.read()
    found_resume = False
    words = queue.Queue()
    for word in raw_words.split():
        if resume is not None:
            if found_resume:
                extend_words(word)
            elif word == resume:
                found_resume = True
                print(f'Resumeing wordlist from: {resume}')
        else:
            print(word)
            extend_words(word)

    return words


def dir_bruter2(words):
    print("----------dir words2---------")
    headers = {'User-Agent': AGENT}
    while not words.empty():
        url = f'{TARGET}{words.get()}'
        print(url)
        try:
            r = requests.get(url, headers=headers)
            print(r.status_code)
        except Exception as err:
            print(f"{err}")
            
def dir_bruter(words):
    print("----------dir words---------")
    headers = {'User-Agent': AGENT}
    while not words.empty():
        url = f'{TARGET}{words.get()}'
#        print(url)
        try:
            r = requests.get(url, headers=headers)
#            print(r.status_code)
        except:
            sys.stderr.write('x')
            sys.stderr.flush()
            continue

        if r.status_code == 200:
            print(f'\nSuccess ({r.status_code}: {url})')
        elif r.status_code == 404:
            sys.stderr.write('.');sys.stderr.flush()
        else:
            print(f'{r.status_code}=>{url}')
                
    
if __name__ == '__main__':
    words = get_words()
    print(words)
    print('Press return to continue.')
    sys.stdin.readline()
    
    #words2 = queue.Queue()
    #words2.put(f'/CVS/')
    
    for _ in range(THREADS):
        #t = threading.Thread(target=dir_bruter2, args=(words2,))
        t = threading.Thread(target=dir_bruter, args=(words,))        
        t.start()

