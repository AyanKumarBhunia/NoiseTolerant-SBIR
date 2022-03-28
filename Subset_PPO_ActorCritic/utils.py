import random
import string
import time
import pickle


def random_string(n):
    return ''.join(random.choice('0123456789abcdefghijklmnopqrstuvwxyz') for i in range(n))

def id_random_string(name=None):
    N = 7
    res = str(name)
    if name == None:
        res = ''.join(random.choices(string.ascii_lowercase + string.digits, k = N))
    return 'imgs/' + res + '.png'