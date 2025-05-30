from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(f"handle starttag => tag is {tag}")

    def handle_data(self, data):
        print(f"handle starttag => data is {data}")

    def handle_endtag(self, tag):
        print(f"handle endtag => tag is {tag}")        


parser = MyHTMLParser()
parser.feed("<title>Python rock!</title><h1>Mike Mike</h1>")
