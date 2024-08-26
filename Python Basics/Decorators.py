#decorators in python are used to modify the behavior of functions 

#example
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called")
        func()
        print("Something is happening after the function is called")
    return wrapper

def say_hello():
    print("Hello!")

#a more famous example is the @ symbol
#this is used to decorate a function with a decorator
@my_decorator
def say_goodbye():
    print("Goodbye!")

say_hello()
say_goodbye()

#decorators can also take arguments

def my_decorator2(func):
    def wrapper(*args,**kwargs):
        print("Something is happening before the function is called")
        func(*args,**kwargs)
        print("Something is happening after the function is called")
    return wrapper

@my_decorator2
def say_something(something):
    print(something)

say_something("Hello!")