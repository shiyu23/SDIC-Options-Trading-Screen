import sys
import time
import zmq
import threading
import json
import re

sys.coinit_flags = 0
global QUOTE_CONNECT_SUCCESS
QUOTE_CONNECT_SUCCESS=False
global TRADE_CONNECT_SUCCESS
TRADE_CONNECT_SUCCESS=False
global userdict
userdict = {}
init_login_obj = {"Request":"LOGIN","Param":{"SystemName":"ZMQ",
        "ServiceKey":"8076c9867a372d2a9a814ae710c256e2"}}
class tcore_zmq():
    def __init__(self):
        self.context = zmq.Context()

    def trade_connect(self,port,login_obj = init_login_obj):
        self.tsocket = self.context.socket(zmq.REQ)
        self.tsocket.connect("tcp://127.0.0.1:%s" % port)
        self.tsocket.send_string(json.dumps(login_obj))
        message = self.tsocket.recv()
        message = message[:-1]
        data = json.loads(message)
        return data

    def quote_connect(self,port,login_obj = init_login_obj):
        self.qsocket = self.context.socket(zmq.REQ)
        self.qsocket.connect("tcp://127.0.0.1:%s" % port)
        self.qsocket.send_string(json.dumps(login_obj))
        message = self.qsocket.recv()
        message = message[:-1]
        data = json.loads(message)
        return data

    def trade_logout(self,key):
        obj = {"Request":"LOGOUT","SessionKey":key}
        self.tsocket.send_string(json.dumps(obj))
        return

    def quote_logout(self,key):
        obj = {"Request":"LOGOUT","SessionKey":key}
        self.qsocket.send_string(json.dumps(obj))
        return

    def account_lookup(self,key):
        obj = {"Request":"ACCOUNTS","SessionKey":key}
        self.tsocket.send_string(json.dumps(obj))
        message = self.tsocket.recv()[:-1]
        data = json.loads(message)
        return data

    def restore_report(self,key,page):
        obj = {"Request":"RESTOREREPORT","SessionKey":key,"QryIndex":page}
        self.tsocket.send_string(json.dumps(obj))
        message = self.tsocket.recv()[:-1]
        data = json.loads(message)
        return data

    def new_order(self,key,Param):
        obj = {"Request":"NEWORDER","SessionKey":key}
        obj["Param"] = Param
        self.tsocket.send_string(json.dumps(obj))
        message = self.tsocket.recv()[:-1]
        data = json.loads(message)
        return data

    def replace_order(self,key,Param):
        obj = {"Request":"REPLACEORDER","SessionKey":key}
        obj["Param"] = Param
        self.tsocket.send_string(json.dumps(obj))
        message = self.tsocket.recv()[:-1]
        data = json.loads(message)
        return data

    def cancel_order(self,key,Param):
        obj = {"Request":"CANCELORDER","SessionKey":key}
        obj["Param"] = Param
        self.tsocket.send_string(json.dumps(obj))
        message = self.tsocket.recv()[:-1]
        data = json.loads(message)
        return data

    def margin(self,key,AM):
        obj = {"Request":"MARGINS","SessionKey":key,"AccountMask":AM}
        self.tsocket.send_string(json.dumps(obj))
        message = self.tsocket.recv()[:-1]
        data = json.loads(message)
        return data

    def position(self,key,AM,page):
        obj = {"Request":"POSITIONS","SessionKey":key,"AccountMask":AM,"QryIndex":page}
        self.tsocket.send_string(json.dumps(obj))
        message = self.tsocket.recv()[:-1]
        data = json.loads(message)
        return data

    def subquote(self,key,Param):
        obj = {"Request":"SUBQUOTE","SessionKey":key}
        obj["Param"] = Param
        self.qsocket.send_string(json.dumps(obj))
        message = self.qsocket.recv()[:-1]
        data = json.loads(message)
        return data

    def unsubquote(self,key,Param):
        obj = {"Request":"UNSUBQUOTE","SessionKey":key}
        obj["Param"] = Param
        self.qsocket.send_string(json.dumps(obj))
        message = self.qsocket.recv()[:-1]
        data = json.loads(message)
        return data

    def sub_history(self,key,Param):
        obj = {"Request":"SUBQUOTE","SessionKey":key}
        obj["Param"] = Param
        self.qsocket.send_string(json.dumps(obj))
        message = self.qsocket.recv()[:-1]
        data = json.loads(message)
        return data  

    def get_history(self,key,Param):
        obj = {"Request":"GETHISDATA","SessionKey":key}
        obj["Param"] = Param
        self.qsocket.send_string(json.dumps(obj))
        message = (self.qsocket.recv()[:-1]).decode("utf-8")
        index =  re.search(":",message).span()[1]  # filter 
        symbol = message[:index-1]
        message = message[index:]
        message = json.loads(message)
        return message

    def QueryInstrumentInfo(self, key, sym):
        obj = {"Request" : "QUERYINSTRUMENTINFO" , "SessionKey" : key , "Symbol" : sym}
        self.qsocket.send_string(json.dumps(obj))
        message = self.qsocket.recv()[:-1]
        data = json.loads(message)
        return data

    def QueryAllInstrumentInfo(self, key, type):
        obj = {"Request": "QUERYALLINSTRUMENT", "SessionKey": key, "Type": type}
        self.qsocket.send_string(json.dumps(obj))
        message = self.qsocket.recv()[:-1]
        data = json.loads(message)
        return data

    def TradePong(self,key):
        obj = {"Request":"PONG","SessionKey":key}
        self.tsocket.send_string(json.dumps(obj))
        message = self.tsocket.recv()[:-1]
        data = json.loads(message)
        return data

    def QuotePong(self,key):
        obj = {"Request":"PONG","SessionKey":key}
        self.qsocket.send_string(json.dumps(obj))
        message = self.qsocket.recv()[:-1]
        data = json.loads(message)
        return data