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

#lambdas are useful when making custom comparators for sorting
#example
points2D = [(1,2),(15,1),(5,-1),(10,4)]
points2D_sorted = sorted(points2D,key=lambda x:x[1])
print(points2D_sorted)
