# %%
def recurs(j):
    if j == 1:
        return 3/13
    if j == 3:
        return((1/6) + (4/5) * recurs(2))
    if j == 2:
        return((1/3) * (5/9) + (2/3) * recurs(1))

print(f'{recurs(1)}, {recurs(2)}, {recurs(3)} and sum {recurs(1) + recurs(2) + recurs(3)}')
# %%
def recurs(j):
    if j == 3:
        return((1/6) + (1/6) * recurs(2) + (1/3) * recurs(1))
    if j == 2:
        return((1/3) * (5/9) + (1/2) * recurs(1))
    if j == 1:
        return 3/13
print(f'{recurs(1)}, {recurs(2)}, {recurs(3)} and sum {recurs(1) + recurs(2) + recurs(3)}')

# %%
def recurs(j):
    if j == 1:
        return 3/13
    if j == 3:
        return (1/6) + (4/5) * recurs(2) + (1/2) *
    if j == 2:
        return (1/3) * (5/9) + (2/3) * recurs(1)

print(f'{recurs(1)}, {recurs(2)}, {recurs(3)} and sum {recurs(1) + recurs(2) + recurs(3)}')

# %%
def test(j):
    if j == 1:
        return((1/2) * (6/13))
    if j == 2:
        return((1/2) * (4/13) + (1/3) * (5/9))
    if j == 3:
        return((1/2) * (3/13) + (1/3) * (4/9) + (1/6) * 1)

# %%
print(f'{test(1)}, {test(2)}, {test(3)} and sum {test(1) + test(2) + test(3)}')

# %%
def prod_program(n,j):
    a = 1
    for k in range(1, n-j + 1):
        a *= (1/2) + (1/2) * j / (n - k + 1)
    return(a)
# %%
prod_program(3, 2) - prod_program(3,1)
# %%
