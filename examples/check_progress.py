import pickle
import numpy as np

with open('success_dict.pkl', 'rb') as f:
    successful = pickle.load(f)
with open('costs_dict.pkl', 'rb') as f:
    costs = pickle.load(f)
print(successful)
print(costs)

for name, key in successful.items():
    print(f'Success rates for {name}: {sum(key)/len(key)}')
    print(f'Average cpst if landed for {name}: {np.dot(np.array(key),np.array(costs[name]))/sum(key)}')