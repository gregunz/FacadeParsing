def to_square_crops(x, y, size):
    assert x % size == 0
    assert y % size == 0
    crops = []
    for i in range(x // size):
        for j in range(y // size):
            crops.append((i * size, j * size))
    return crops


def to_multiple_of_shape(x, y, m, max_size):
    n_x = x // m
    n_y = y // m
    n_max_size = max_size // m

    if n_x > n_max_size and n_x > n_y:
        n_x = n_max_size
        n_y = round(y * n_max_size / x)
    else:
        n_y = n_max_size
        n_x = round(x * n_max_size / y)

    return (n_x * m, n_y * m)
