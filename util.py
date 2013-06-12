def argmax(elements, function):
    """Return an element with highest function(elements[i]) score; tie goes to first one.
    >>> argmax(['one', 'to', 'three'], len)
    'three'
    """
    best_element = None
    best_score = -9e308
    for element in elements:
        score = function(element)
        if score > best_score:
            best_score = score
            best_element = element
    return best_element


if __name__ == '__main__':
    import doctest
    doctest.testmod()