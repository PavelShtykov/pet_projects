def validation(val, *funcs):
    curr_val = val
    for func in funcs:
        curr_val, stop = func(curr_val)
        if stop:
            return curr_val

    return curr_val


def is_int():
    def return_func(val):
        try:
            res = int(val)
        except Exception:
            raise TypeError('Введите целое число')

        return res, False

    return return_func


def is_float():
    def return_func(val):
        try:
            res = float(val)
        except Exception:
            raise TypeError('Введите целое число')

        return res, False

    return return_func


def in_range(start, end, left, right):
    def return_func(val):
        if left == '[' and right == ']':
            if start <= val <= end:
                return val, False
        if left == '[' and right == ')':
            if start <= val < end:
                return val, False
        if left == '(' and right == ']':
            if start < val <= end:
                return val, False
        if left == '(' and right == ')':
            if start < val < end:
                return val, False

        raise TypeError('Введите число в диапозоне ' + left + f'{start}, {end}' + right)

    return return_func


def is_str(str_map):
    def return_func(val):
        if val in str_map:
            return str_map[val], True

        return val, False

    return return_func
