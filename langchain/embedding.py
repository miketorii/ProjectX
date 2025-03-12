from langchain_openai import AzureOpenAIEmbeddings

print("---------------------------------start---------------------------------")

embeddings = AzureOpenAIEmbeddings(
    model="my-text-embedding-3-large",
    #azure_endpoint=AZURE_OPENAI_EMBEDDING_ENDPOINT,
    #api_key=AZURE_OPENAI_EMBEDDING_API_KEY,
    # openai_api_version=AZURE_OPENAI_EMBEDDING_API_VERSION    
    # dimensions: Optional[int] = None, # Can specify dimensions with new text-embedding-3 models
    # azure_endpoint="https://<your-endpoint>.openai.azure.com/", If not provided, will read env variable AZURE_OPENAI_ENDPOINT
    # api_key=... # Can provide an API key directly. If missing read env variable AZURE_OPENAI_API_KEY
    # openai_api_version=..., # If not provided, will read env variable AZURE_OPENAI_API_VERSION
)

query = "AWSのS3からデータを読み込むためのDocument loaderはありますか？"

vector = embeddings.embed_query(query)

print(len(vector))
print(vector)

print("---------------------------------end---------------------------------")
