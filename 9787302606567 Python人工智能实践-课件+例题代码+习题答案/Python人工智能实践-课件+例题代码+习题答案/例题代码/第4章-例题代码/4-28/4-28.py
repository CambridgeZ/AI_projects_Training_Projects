import matplotlib.pyplot as plt

def read_txt(file):
    with open(file, 'r') as temp:
        ls = [x.strip().split() for x in temp]
    return ls

def plot_line(ls):
    x = [int(x[0]) for x in ls]
    higher = [int(x[1]) for x in ls]
    lower = [int(x[2]) for x in ls]

    plt.plot(x, higher, marker='o', color='r')
    plt.plot(x, lower, marker='*', color='b')
    plt.xticks(list(range(1, 32,2)))
    plt.yticks(list(range(-10, 30, 5)))
    plt.axhline(0, linestyle='--', color='g')
    plt.show()
    plt.savefig('temp_curve.png')

filename = 'temp.txt'
temp_list = read_txt(filename)
plot_line(temp_list)
