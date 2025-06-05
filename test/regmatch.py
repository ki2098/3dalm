import re

def parse_formula(str):
    number_pattern = r'([+-]?\d*\.?\d*(?:[eE][+-]?\d+)?)'
    pattern = rf'{number_pattern}sin\({number_pattern}t{number_pattern}\){number_pattern}'

    match = re.match(pattern, str)

    a = match.group(1)
    b = match.group(2)
    c = match.group(3)
    d = match.group(4)

    if a == "" or a == "+":
        a = 1.0
    elif a == "-":
        a = -1.0
    else:
        a = float(a)

    if b == "" or b == "+":
        b = 1.0
    elif b == "-":
        b = -1.0
    else:
        b = float(b)

    if c == "":
        c = 0.0
    else:
        c = float(c)

    if d == "":
        d = 0.0
    else:
        d = float(d)

    print(f'{a}sin({b}t+{c})+{d}')

while True:
    str = input("input formula: ")
    if str == 'q':
        break
    parse_formula(str)