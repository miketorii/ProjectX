#codeing: UTF-8

import sys
import io
import logging
import json
import decimal
import time

import datetime
from datetime import datetime as dt

import boto3
from boto3.session import Session
from boto3.dynamodb.conditions import Key, Attr

#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
logging.basicConfig(level=logging.INFO, filename=(__file__ + ".log"), format="%(asctime)s %(levelname)s %(filename)s %(lineno)d %(funcName)s | %(message)s")

accesskey = "xxx"
secretkey = "xxxxxxx"
region    = "localhost:8000"
#region    = "ap-northeast-1"

tmp_today = datetime.datetime.today()
print tmp_today

session = Session(aws_access_key_id=accesskey, aws_secret_access_key=secretkey, region_name=region)
#dynamodb = session.resource('dynamodb')

dynamodb = boto3.resource('dynamodb', endpoint_url='http://localhost:8000')

#####################################################
#  
#####################################################



#####################################################
# Create Table
#####################################################
def MoviesCreateTable():
    try:
        print "MoviesCreateTable"
        logging.info("<<<<< %s Start >>>>>",__name__)

        table = dynamodb.create_table(
            TableName='Movies',
            KeySchema=[
                {
                    'AttributeName': 'year',
                    'KeyType': 'HASH'
                },
                {
                    'AttributeName': 'title',
                    'KeyType': 'RANGE'
                },
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': 'year',
                    'AttributeType': 'N'
                },
                {
                    'AttributeName': 'title',
                    'AttributeType': 'S'
                },
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 10,
                'WriteCapacityUnits': 10
            }
        )
    except Exception as e:
        print "Exception .."
        print type(e)
        print e
        logging.error("Type : %s", type(e))
        logging.error(e)

#    print __name__ ,"--End--"
    logging.info("<<<<< %s End >>>>>",__name__)


#####################################################
# Load Json Data
#####################################################
def MoviesLoadData():
    try:
        logging.info("<<<<< %s Start >>>>>",__name__)
        print "MoviesLoadData"
        with open("moviedata.json") as json_file:
            movies = json.load(json_file, parse_float = decimal.Decimal)
            for movie in movies:
                year = int(movie['year'])
                title = movie['title']
                info = movie['info']
                table.put_item(
                    Item={
                        'year': year,
                        'title': title,
                        'info': info,
                    }
                )
#                print year
#                print title
#                print info
    except Exception as e:
        print "Exception .."
        print type(e)
        print e
        logging.error("Type : %s", type(e))
        logging.error(e)

    print __name__ ,"--End--"
    logging.info("<<<<< %s End >>>>>",__name__)

#####################################################
# Delete Table
#####################################################
def MoviesDeleteTable():
    try:
        print "MoviesDeleteTable"
        table = dynamodb.Table('Movies')
        table.delete()
    except Exception as e:
        print "Exception .."
        print type(e)

#####################################################
# Query Table
#####################################################
def MoviesQuery01():
    try:
        logging.info("<<<<< %s Start >>>>>",__name__)
        print __name__

        table = dynamodb.Table('Movies')
        print "Movies from 1933"

        response = table.query(
            KeyConditionExpression=Key('year').eq(1933)
        )
        for i in response['Items']:
            logging.info("%s : %s", i['year'], i['title'])
            print i['year'], i['title']

    except Exception as e:
        print "Exception .."
        print type(e)
        print e
        logging.error("Type : %s", type(e))
        logging.error(e)

#    print __name__ ,"--End--"
    logging.info("<<<<< %s End >>>>>",__name__)

#####################################################
#  main
#####################################################

if __name__ == '__main__':
    print "------start------"

    table = dynamodb.Table('Movies')
    print table

    # Create Table
#    MoviesCreateTable()

    # Load Json Data
#    MoviesLoadData()

    # Query
    MoviesQuery01()

    # Delete Table
#    MoviesDeleteTable()

    print "-------end-------"

