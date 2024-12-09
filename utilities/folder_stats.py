import os
import sys

def getOffset(count, max = 19):
    offset1 = max-len(c)
    if (count < 1000):
        offset1 = offset1 + 1
    if (count < 100):
        offset1 = offset1 + 1
    if (count < 10):
        offset1 = offset1 + 1
    return offset1

print('|------------ folder stats ------------|\n')

print('|=====      training images       =====|')
print('|---------------- objects -------------|')
categories = ['buses', 'cars', 'pedestrians', 'traffic_signs']
object_count = 0
for c in categories:
    count = len(os.listdir(f"data/training/{c}"))
    object_count = object_count + count;
    offset1 = getOffset(count)
    print(f'    {c}: {" " * offset1}{count}')
training_count = object_count
offset1 = getOffset(object_count, 25)
print(f'    objects: {" " * offset1}{object_count}')

print('|---------------- weather -------------|')
categories = ['rainy','snowy','night','cloudy','sunny']
weather_count = 0
for c in categories:
    count = len(os.listdir(f"data/training/{c}"))
    weather_count = weather_count + count;
    offset1 = getOffset(count, 19)
    print(f'    {c}: {" " * offset1}{count}')
training_count = training_count + weather_count
offset1 = getOffset(weather_count, 17)
print(f'    weather: {" " * offset1}{weather_count}')
offset1 = getOffset(training_count, 16)
print(f'    training: {" " * offset1}{training_count}')
    
print('\n|=====      testing  images       =====|')
print('|---------------- objects -------------|')
categories = ['buses', 'cars', 'pedestrians', 'traffic_signs']
object_count = 0
for c in categories:
    count = len(os.listdir(f"data/testing/{c}"))
    object_count = object_count + count;
    offset1 = getOffset(count, 19)
    print(f'    {c}: {" " * offset1}{count}')
testing_count = object_count
offset1 = getOffset(object_count, 25)
print(f'    objects: {" " * offset1}{object_count}')

print('|---------------- weather -------------|')
categories = ['rainy','snowy','night','cloudy','sunny']
weather_count = 0
for c in categories:
    count = len(os.listdir(f"data/testing/{c}"))
    weather_count = weather_count + count;
    offset1 = getOffset(count, 19)
    print(f'    {c}: {" " * offset1}{count}')
testing_count = testing_count + weather_count
offset1 = getOffset(weather_count, 17)
print(f'    weather: {" " * offset1}{weather_count}')
offset1 = getOffset(testing_count, 17)
print(f'    testing: {" " * offset1}{testing_count}')
    
print('|------------ folder stats ------------|')
