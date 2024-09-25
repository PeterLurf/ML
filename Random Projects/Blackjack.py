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