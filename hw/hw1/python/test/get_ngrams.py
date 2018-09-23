from itertools import accumulate

def trim_tuple_head(t, n):
    assert isinstance(t, tuple)
    assert isinstance(n, int)

    if (len(t) > n):
        return t[(len(t) - n):]
    else:
        return t

def fill_with_starts_at_beginning(t,n):
    assert isinstance(t, tuple)

    return (("START",)*(n-len(t))) + t

def get_ngrams_1(sequence, n):
    assert isinstance(sequence, list)
    assert isinstance(n, int) and n >= 1

    wrapped_in_tuples = map(lambda x: (x,), sequence)
    duplicated = accumulate(wrapped_in_tuples, lambda tuple1, tuple2: tuple1 + tuple2)
    filled_with_starts = map(lambda t: fill_with_starts_at_beginning(t,n), duplicated)
    trimed = list(map(lambda t: trim_tuple_head(t,n), filled_with_starts))
    stop_tuple = trim_tuple_head(trimed[-1] + ("STOP",), n)
    if (n > 1):
        return trimed + [stop_tuple]
    else:
        return [("START",)] + trimed + [stop_tuple]

def get_ngrams(sequence, n):
    result = []
    length = len(sequence)
    if n > 1:
        for i in range(length):
            if i > 0:
                result.append(result[i-1][-(n-1):] + (sequence[i],))
            else:
                result.append((sequence[0],))
        for i in range(length):
            result[i] = (("START",)*(n - i - 1) + result[i])
        result.append(result[length-1][-(n-1):] + ("STOP",))
    else:
        for word in sequence:
            result.append((word,))
        result.append(("STOP",))

    if n == 1:
        result = [("START",)] + result

    return result
