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
def create_items():
    print("---create_items---")

    
print("----------------------start--------------------")

create_items()

credential = DefaultAzureCredential()

endpoint = config.settings['host']
print(endpoint)

client = CosmosClient(endpoint, credential)

#https://qiita.com/baku2san/items/85714f90c094cebedb44

print("----------------------end--------------------")
