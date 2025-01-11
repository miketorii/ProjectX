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
    device = {'id': device_id,
              'deviceid': device_id,
              'modelname': 'iR-ADV C550'+str(num),
              'ipaddr': '192.168.11.'+str(addr4),
              'location': 'building H '+str(num)+'F',
              'options': [
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
def delete_item(container, pkey):
    print("---delete_items---")
    response = container.delete_item(item=pkey, partition_key=pkey)
    print(response)

######################################
#
def read_item(container, pkey):
    print("---read_items---")
    response = container.read_item(item=pkey, partition_key=pkey)
#    print(response)

    return response
   
######################################
#
def execute_db_ops():
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

######################################
#
def display_device(device):
    print(device.get('id'))
    print(device.get('deviceid'))    
    print(device.get('ipaddr'))
    print(device.get('location'))    
    print(device.get('status'))
    print(device.get('number'))
    print(device['options'][0].get('adf'))
    print(device['options'][0].get('paper_deck'))
    print(device['options'][0].get('ps'))
    print(device['options'][0].get('finisher'))        
    
                  
######################################
#
def query_devices():
    print("----------------------start--------------------")
    
    credential = DefaultAzureCredential()
    
    endpoint = config.settings['host']
    print(endpoint)    
    client = CosmosClient(endpoint, credential)
    
    databasename = config.settings['database_id']
    db = client.get_database_client(databasename)

    containername = config.settings['container_id']
    container = db.get_container_client(containername)
    
#    devices = query_items(container)
#    print(devices)

#    delete_item(container, "dcab0009")

    device = read_item(container, "dcab0004")
    print(device)

    display_device(device)

    devices = query_items(container)
    print(devices)
    
    print("----------------------end--------------------")
    
######################################
#    
if __name__ == '__main__':
    #execute_db_ops()

    query_devices()
    
