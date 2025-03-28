import os
from dotenv import load_dotenv

class Settings():
    
    def readenv(self):
        load_dotenv(".env")
        
        #print( os.environ )
        print( os.environ["AZURE_OPENAI_EMBEDDED_API_KEY"] )
        print( os.environ["AZURE_OPENAI_EMBEDDED_ENDPOINT"] )
        print( os.environ["AZURE_OPENAI_EMBEDDED_API_VERSION"] )        
        

if __name__ == "__main__":
    conf = Settings()
    conf.readenv()
    
