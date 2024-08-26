def matrix_dot_vector(a:list[list[int|float]],b:list[int|float])-> list[int|float]:
    if len(b) != len(a[0]):
        return -1
    
    result = []
    for i in range(len(a)):
        temp = 0
        for j in range(len(b)):
            temp += a[i][j] * b[j]
        result.append(temp)
    return result

