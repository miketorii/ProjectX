import urllib.parse
import urllib.request

url = 'https://www.google.com'

def runget():
    print("------run get------")

    with urllib.request.urlopen(url) as response:
        content = response.read()
        print(content)
        
if __name__ == "__main__":
    runget()
    
    
