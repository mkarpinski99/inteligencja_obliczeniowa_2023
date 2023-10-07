import math


def prime(n: int) -> bool:
    if n <= 1:
        return False
    if n == 2:
        return True
    for i in range(2, int(math.ceil(math.sqrt(n))) + 1):
        if n % i == 0:
            return False
    return True


def select_primes(x: list) -> list:
    return list(filter(prime, x))
