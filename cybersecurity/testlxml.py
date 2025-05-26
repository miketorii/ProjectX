from io import BytesIO
from lxml import etree
import requests

url = 'https://www.google.com'
r = requests.get(url)

content = r.content

parser = etree.HTMLParser()
content2=etree.parse(BytesIO(content), parser=parser)

for link in content2.findall('.//a'):
    print(f"{link.get('href')} -> {link.text}")

    
