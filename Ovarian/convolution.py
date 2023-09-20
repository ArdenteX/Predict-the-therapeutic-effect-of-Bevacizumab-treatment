import numpy as np

x = np.array([1, 2, 5])
y = np.array([2, 4, 9])

con = []
for i in range(y.size):
    if i == 0:
        con.extend(x*y[i])
        print("{} ---- >{}".format(i + 1, x * y[i]))
        continue

    tmp = x*y[i]
    print("{} ---- >{}".format(i + 1, tmp))
    for j in range(tmp.size):
        start = j+i
        try:
            con[start] = con[start] + tmp[j]

        except IndexError:
            con.append(tmp[j])

