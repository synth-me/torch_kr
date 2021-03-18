import os
import subprocess
from korean_romanizer.romanizer import Romanizer
import jamotools
import pandas as pd
import sys
import numpy as np
import jamo
from jamo import j2h
import re
import requests
import bs4
from bs4 import BeautifulSoup
import time
import hangul_romanize
from hangul_romanize import Transliter
from hangul_romanize.rule import academic
import json
import subprocess

def file_name_def(file:str)->str:
# here we get the absolute path to acess the files
# we need to write down the information
    x = os.path.realpath(__file__)
    y = x.split("\\")
    y.remove(y[len(y)-1])
    p = "\\".join(y)
    return p+r"\{z}".format(z=file)

def normalizer(value:list,maximum)->str:
# here we normalize the list with a maximum of
# the length
    if len(value) < maximum :
        while len(value) < maximum :
            value.append(0)
    return "done"

def encode_strings(string:list,v)->dict:
# here we encode the string in binary strings
# and then back in numbers
    d = {}
    for i in range(len(string)) :
        vector = list(v.vectorize(u"{x}".format(x=string[i])))
        try:
            r = Romanizer(string[i]).romanize()
        except:
            transliter = Transliter(academic)
            transliter.translit(string[i])
        normalizer(vector,7)
        res = list(format(ord(i), '08b') for i in r)
        number = list(map(lambda w :(int(w, 2)),res))
        normalizer(number,7)
        double = "{hangul}/{roman}".format(hangul=string[i],roman=r)
        d[double] = ((vector,number))
    return d

def encode_unique(string:str,v)->list:
# here we encode unique strings into numbers
    vector = list(v.vectorize(u"{x}".format(x=string)))
    normalizer(vector,7)
    return vector

def decode_vector(binary:list)->str:
# here we deocdify from the integer number to
# the binary and then back to the string
    string = []
    for j in binary :
        j = int(j)
        by = f'{j:08b}'
        binary_int = int(by, 2)
        byte_number = binary_int.bit_length() + 7 // 8
        binary_array = binary_int.to_bytes(byte_number, "big")
        ascii_text = binary_array.decode('utf-8')
        n = ''.join(char if char.isprintable() else '' for char in ascii_text)
        string.append(n)
    return ''.join(string)

def generate_train_set():
# here we scrap wikipedia's unicode table
# to find all the possible jamo combinations
    mtch = []
    page = requests.get("https://en.wikipedia.org/wiki/Hangul_Syllables").text
    td = BeautifulSoup(page,'lxml')
    x = re.compile(r"[\uAC00-\uD7A3]")
    p = td.find_all('td')
    for j in p:
        m = str(j).split()
        if len(m) == 1 :
            n = x.search(u'{m}'.format(m=j))
            if n != None:
                mtch.append(n.group(0))
                yield n.group(0)
        else:
            print("loading...")
            pass


def main():

    rounds = int(input("rounds: "))
    save = True

    symbol_list = []
    symbol_string = []
    symbol_new = []
    s = generate_train_set()
    v = jamotools.Vectorizationer(rule=jamotools.rules.RULE_1,max_length=3,prefix_padding_size=0)
    for y in range(rounds):
        symbol = next(s)
        symbol_list.append(symbol)
    symbol_eval = encode_strings(symbol_list,v)

    if save != False :
        symbol_dict = {"train_data":[],"predict_data":[]}
        for zvalue in symbol_eval:
            template = {}
            converted_inpt = list(map(int,symbol_eval[zvalue][0]))
            converted_outpt = list(map(int,symbol_eval[zvalue][1]))
            template["input"] = converted_inpt
            template["output"] = converted_outpt
            symbol_dict["train_data"].append(template)

        for value in symbol_eval:
            symbol_string.append({"input:":list(jamotools.split_syllable_char(value.split("/")[0])),"output":value.split("/")[1]})

        while True :
            inpt = input("input: ")
            if inpt == "f" :
                break
            else:
                new_l = list(map(int,encode_unique(inpt,v)))
                template = {"input":new_l}
                symbol_dict["predict_data"].append(template)

        with open(file_name_def('train_data.json'),'w') as JsonFile :
            j = json.dump(symbol_string,JsonFile)

        print("****saved****")

    data_frame = pd.DataFrame(symbol_eval,index=['input','output'])
    return data_frame
