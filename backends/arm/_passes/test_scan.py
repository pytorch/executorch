import torch

# Standard scan implementation
# https://docs.pytorch.org/xla/release/r2.6/features/scan.html
def scan(combine_fn, init, xs):
    ys = []
    carry = init
    for x in xs:
        carry, y = combine_fn(carry, x)
        ys.append(y)
    return carry, torch.stack(ys, dim=0)

# Scan implementation with tosa.WHILE_LOOP
def while_scan(combine_fn, init, xs):
    def while_body(counter, xs, carry, ys):
        x = xs[counter]
        carry, y = combine_fn(carry, x)
        ys[counter] = y
        return counter + 1, xs, carry, ys
    def cond(counter, xs, carry):
        return counter<len(xs)
        
    counter = 0
    carry = init
    # Shape of ys needs to be calculated in a smarter way, but should be known from combine_fn
    ys = torch.empty_like(((xs)))
    while cond(counter, xs, carry):
        counter, xs, carry, ys = while_body(counter,xs,carry, ys)

    return carry, ys.unsqueeze(-1)

xs = torch.arange(1, 5)
init = torch.tensor((2,))

# r = (tensor(48), tensor([2, 4, 12, 48]))
r1 = scan(lambda x, y: (x * y, x * y), init, xs)
r2 = while_scan(lambda x, y: (x * y, x * y), init, xs)

print(r1)
print(r2)
