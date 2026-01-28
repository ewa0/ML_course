import numpy as np

x = 7 ** 4
print("power:", x)

s = "Hi there Sam!"
print("\nsplit:", s.split())

planet = "Earth"
diameter = 12742

string = "The diameter of {} is {} km.".format(planet, diameter)
print("\nformat method:", string)

lst = [1, 2, [3, 4], [5, [100, 200, ['hello']], 23, 11], 1, 7]
print("\nnested list accessing:", lst[3][1][2])

d = {'k1': [1, 2, 3, {'tricky': ['oh', 'man', 'inception', {'target': [1, 2, 3, 'HELLO']}]}]}
print("\nnested dictionary accessing:", d["k1"][3]["tricky"][3]["target"][3])

email = "user@domain.com"
print("\nemail website domain:", email.split("@")[-1])

findDog = "Is there a dog here?"
is_there = lambda str: "dog" in str
print("\nDoes the string:", findDog, "contain word <dog>?:", is_there(findDog))

dogs_string = "This dog runs faster than the other dog dude!"
print("\nHow many times <dog> occurs in the string:", dogs_string, "?")
print(dogs_string.count("dog"))

seq = ['soup', 'dog', 'salad', 'cat', 'great']
s_words = list(filter(lambda word: word.startswith("s"), seq))
print("\nwords beginning with s in:", seq, "are:", s_words)


def caught_speeding(speed, is_birthday):
    if is_birthday:
        velocity = speed - 5
    else:
        velocity = speed

    if velocity <= 60:
        return "No Ticket"

    elif 60<velocity<=80:
        return "Small Ticket"

    elif velocity>80:
        return "Big Ticket"
    else:
        return "No Ticket"


s =82
b = True
print("\nPolice officer says speed was {}. Is it my birthday? {} \nTicket?".format(s, b), caught_speeding(s, b))
