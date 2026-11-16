import random

numbers = [10, 20, 30, 40, 50, 60, 70, 80]
numbers.extend([200, 300])
print('After addition:', numbers)
for value in [10, 30]:
    numbers.remove(value)
print('After removal:', numbers)
print('Ascending:', sorted(numbers))
print('Descending:', sorted(numbers, reverse=True))

scores = (45, 89.5, 76, 45.4, 89, 92, 58, 45)
print('Highest and index:', max(scores), scores.index(max(scores)))
print('Lowest and count:', min(scores), scores.count(min(scores)))
print('Reversed list:', list(reversed(scores)))
score = float(input('Enter score to search [76]: ') or '76')
print('First occurrence index:', scores.index(score) if score in scores else 'Not present')

values = [random.randint(100, 900) for _ in range(100)]
def prime(n):
    return n > 1 and all(n % i for i in range(2, int(n ** 0.5) + 1))
odds = [n for n in values if n % 2]
evens = [n for n in values if n % 2 == 0]
primes = [n for n in values if prime(n)]
print('Random numbers:', values)
for name, group in [('Odd', odds), ('Even', evens), ('Prime', primes)]:
    print(name, 'count:', len(group), 'numbers:', group)

A = {34, 56, 78, 90}
B = {78, 45, 90, 23}
print('Union:', A | B)
print('Intersection:', A & B)
print('Symmetric difference:', A ^ B)
print('A subset of B:', A.issubset(B))
print('B superset of A:', B.issuperset(A))
x = int(input('Score to remove from team A [34]: ') or '34')
if x in A:
    A.remove(x)
    print('Updated A:', A)
else:
    print('Score not present in A')

sample_dict = {'name': 'Kelly', 'age': 25, 'salary': 8000, 'city': 'New york'}
sample_dict['location'] = sample_dict.pop('city')
print('Renamed dictionary:', sample_dict)
