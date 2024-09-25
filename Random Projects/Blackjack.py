<<<<<<< HEAD
for i in range (60):
    print(100*0.5*1.246*(i/60 - 0.144)**2)
=======
def isPrime(i):
    if i == 1:
        return False
    for j in range(2, i):
        if i % j == 0:
            return False
    return True

for i in [j**4 + j**2 + 1 for j in range(1, 100)]:
    if isPrime(i):
        print(i)
>>>>>>> 4be3675dee875b5c515a8ff14f17682bded68e98
