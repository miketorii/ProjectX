import base64
import github3
import importlib
import json
import random
import sys
import threading
import time

from datetime import datetime

import os
from dotenv import load_dotenv

def github_connect():
    print("---github connect---")
    user = os.getenv("USER_NAME")
    key = os.getenv("GIT_TOKEN")
    print(user)
    print(key)
    sess = github3.login(token=key)
    print("----repository------")
    repo = sess.repository(user,"ProjectX")
    print(repo)
    get_file_contents("cybersecurity/config","abc.json",repo)

def get_file_contents(dirname, module_name, repo):
    config_json = repo.file_contents(f'{dirname}/{module_name}').content
    print(config_json)
    config = json.loads(base64.b64decode(config_json))
    print(config)
    return config_json
    
def run():
    print("-------run-------")
    github_connect()

if __name__ == "__main__":
    load_dotenv(".env")
    run()
    
