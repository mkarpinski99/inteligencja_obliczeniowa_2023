from math import sqrt
import random

import statistics

list1 = [3, 8, 9, 10, 12]
list2 = [8, 7, 7, 5, 6]

print(f'suma: {[sum(x) for x in zip(list1, list2)]}')
print(f'iloczyn: {[x[0] * x[1] for x in zip(list1, list2)]}')
print(f'iloczyn skalarny: {sum([x[0] * x[1] for x in zip(list1, list2)])}')
print(f'długość wektora1: {sqrt(sum([x**2 for x in list1]))}')
print(f'długość wektora2: {sqrt(sum([x**2 for x in list2]))}')

rand_vector = [random.randint(1, 100) for _ in range(50)]
print('losowy wektor: ', rand_vector)
vec_mean = sum(rand_vector)/len(rand_vector)
print(f'średnia wektora: {vec_mean}')
vec_min = min(rand_vector)
vec_max = max(rand_vector)
print(f'min wektora: {vec_min}')
print(f'max wektora: {vec_max}')
vec_stdev = statistics.stdev(rand_vector)
print(f'odchylenie standardowe: {vec_stdev}')
vec_normalized = [(x-vec_min)/(vec_max-vec_min) for x in rand_vector]
print(f'normalizacja: {vec_normalized}')
print(f'max po normalizacji: {max(vec_normalized)}')
print(f'standaryzacja: {[(x-vec_mean)/vec_stdev for x in rand_vector]}')
print(f'dyskretyzacja: {[[(x - x % 10), (x - x % 10 + 10)] for x in rand_vector]}')
