def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

# Test the functions
x = 15
y = 25
print(f"Addition: {x} + {y} = {add(x, y)}")
print(f"Multiplication: {x} * {y} = {multiply(x, y)}")

# Show it's working
result = add(10, 20) * multiply(2, 3)
print(f"Complex calculation: (10 + 20) * (2 * 3) = {result}")