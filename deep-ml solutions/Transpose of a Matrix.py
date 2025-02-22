def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
	rows = int(len(a)) #2
	cols = int(len(a[0])) #3
	b = []
	for i in range(cols):
		rowTemp = []
		for j in range(rows):
			rowTemp.append(a[j][i])
		b.append(rowTemp)

	return b

#Example
a = [[1,2,3],[4,5,6]]
print(transpose_matrix(a)) # [[1, 4], [2, 5], [3, 6]]
