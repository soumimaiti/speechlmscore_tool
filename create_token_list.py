# Creates tokens.txt

import numpy as np
import os

N_TOK = 50 # Number of tokens
with open("tokens.txt", "w") as f:
    f.write("<blank>\n")
    f.write("<unk>\n")
    for i in range(N_TOK):
        f.write("{}\n".format(i))
    f.write("<sos/eos>\n")
    
