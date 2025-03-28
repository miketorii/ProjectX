import os
from dotenv import load_dotenv

class Settings():
    openai_model: str = "gpt-4o"
    temperature: float = 0.5
    
    def readenv(self):
        load_dotenv(".env")
        
        #print( os.environ )
        print( os.environ["TAVILY_API_KEY"] )
        print( os.environ["OPENAI_API_KEY"] )        
        
