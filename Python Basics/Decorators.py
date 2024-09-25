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

#decorators can also be used to time functions
import time

def time_decorator(func):
    def wrapper(*args,**kwargs):
        start_time = time.time()
        func(*args,**kwargs)
        end_time = time.time()
        print(f"Time taken: {end_time-start_time}")
    return wrapper

@time_decorator
def slow_function():
    time.sleep(2)

slow_function()

#decorators can also be used to cache results in dynamic programming
def cache_decorator(func):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@cache_decorator
def fib(n):
    if n<=1:
        return n
    return fib(n-1) + fib(n-2)

print(fib(35))
