import json
import os
import sys
import uuid

from azure.core.exceptions import AzureError
from azure.cosmos import CosmosClient, PartitionKey

from azure.identity import DefaultAzureCredential

import config

######################################
#
def create_items(container):
    print("---create_items---")

    for i in range(10):
        device = get_device_info(i)
        print(device)
        container.create_item(body=device)

######################################
#   
def get_device_info(num):
    addr4 = 100 + num
    macaddr_base = 0xDCAB0000
    macaddr = macaddr_base + num
    device_id = format(macaddr,'x')
    device = {'deviceid': device_id,
              'modelname': 'iR-ADV C550'+str(num),
              'ipaddr': '192.168.11.'+str(addr4),
              'location': 'building H '+str(num)+'F',
              'optionss': [
                  {'adf': 1,
                   'paper_deck': 1,
                   'ps': 0,
                   'finisher': 2
                   }
              ],
              'status': 10000,
              'number': num
              }

    return device

######################################
#
def query_items(container):
    print("---query_items---")

    items = list(container.query_items(
        query="SELECT * FROM c",
        enable_cross_partition_query=True
    ))

    return items


    
######################################
#    
print("----------------------start--------------------")

credential = DefaultAzureCredential()

endpoint = config.settings['host']
print(endpoint)

client = CosmosClient(endpoint, credential)

databasename = config.settings['database_id']


db = client.get_database_client(databasename)

containername = config.settings['container_id']

container = db.get_container_client(containername)

create_items(container)

devices = query_items(container)
print(devices)

print("----------------------end--------------------")
