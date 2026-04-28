import json
import numpy as np

fjson =  "bunchArrays.json"

with open(fjson, 'r') as fin:
    data = json.load(fin)
fpatb1 = data['schemebeam1']
fpatb2 = data['schemebeam2']
print(f'      filling scheme read, #bunches in b1={np.sum(fpatb1)}, b2={np.sum(fpatb2)}')

mydict = {
    "beam1": fpatb1,  # convert to list
    "beam2": fpatb2
}

# Save to JSON
with open("2025_4x36_converted.json", 'w') as fout:
    json.dump(mydict, fout, indent=4)

print("Saved to mydict.json")
