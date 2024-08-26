#in python lambdas are used to create anonymous functions 
#this means that they are functions without a name and cant be called directly

#lambda arguments: expression 

#example for +1
AddOne = lambda x:x+1
print(AddOne(5))

#example for *2
Mul2 = lambda x:x*2
print(Mul2(5))

#lambdas can be used to generate functions on the fly
def myfunc(n):
    return lambda a:a*n

mydoubler = myfunc(2)
mytripler = myfunc(3)

print(mydoubler(11))
print(mytripler(11))

