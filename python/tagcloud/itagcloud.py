import tagcloud as tc
import importlib
import numpy as np
import time

def reload():
    importlib.reload(tc)

bin_file = "./data/wiki.en.bin"
dict_file = "./american-english"

print("+-----------------------")
print("| import tagcloud as tc")
print("| import numpy as np")
print("|")
print("| use reload() to reload tc")
print("+-----------------------")

print('Loading model')
start_time = time.time()
model = tc.get_model(bin_file)
print("model is loaded in %.2f seconds" % (time.time() - start_time))
print("")

