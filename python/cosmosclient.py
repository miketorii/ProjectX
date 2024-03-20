import azure.cosmos.cosmos_client as cosmos_client
import time
from config import config

class CosmosDBClient():
    def __init__(self):
        print('__init__')
        self.client = cosmos_client.CosmosClient(config['ENDPOINT'], config['PRIMARYKEY'])
        self.database = self.client.get_database_client(config['DATABASE'])
        self.container = self.database.get_container_client(config['CONTAINER'])
#        print(self.database)

    def create_item(self, doc_id, doc_year):
        print('create_item')
        item_pv = self._get_pv(doc_id, doc_year)
        self.container.create_item(body=item_pv)

    def delete_item(self, doc_id, doc_pkey):
        print('delete_item')
        self.container.delete_item(item=doc_id, partition_key=doc_pkey)

    def upsert_item(self, doc_id, doc_pkey):
        print('upsert_item')
        read_item = self.read_item(doc_id, doc_pkey)
        read_item['public'] = 9999999999
        self.container.upsert_item(body=read_item)

    def _get_pv(self, doc_id, doc_year):
        ut = time.time()
        print(ut)
        
        pv = {
            "id": doc_id,
            "date": ut,
            "year": doc_year,
            "office": 2300000000000,
            "public": 30000000000,
            "home_work": 230000000000,
            "home_life": 230000000000
        }
        return pv

    def read_item(self, doc_id, doc_pkey):
        print('read_item')
        item = self.container.read_item(item=doc_id, partition_key=doc_pkey)
        print(item)
        return item

    def read_items(self):
        print('read_items')
        try:
            query = "SELECT * FROM c"
            items = list( self.container.query_items(query=query, enable_cross_partition_query=True) )
            print(items)

        except Exception as e:
            raise e

if __name__ == "__main__":
    print(10*'-'+'cosmos db start'+10*'-')

    client = CosmosDBClient()

    client.read_items()
#    client.read_item("1", 2019)
    client.read_item("2", 2020)

    print(20*'-')

    doc_id = "4"
    doc_year = 2022
    client.create_item(doc_id, doc_year)

    client.read_items()

    print(20*'-')

    client.upsert_item(doc_id, doc_year)
    client.read_item(doc_id, doc_year)

    print(20*'-')

    client.delete_item(doc_id, doc_year)

    client.read_items()

    print(10*'-'+'end'+10*'-')
    