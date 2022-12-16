import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import TextBox
from copy import deepcopy

basis = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

vs = np.array([
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 0],
    [1, 0, 0],

    [0, 1, 1],
    [1, 1, 1],
    [0, 0, 1],
    [1, 0, 1]
])

def rot_x(vs, theta):
    theta = np.radians(theta)
    return vs @ np.array([
        [1,       0,              0,       0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta),  np.cos(theta), 0],
        [0,       0,              0,       1]
    ])

def rot_y(vs, theta):
    theta = np.radians(theta)
    return vs @ np.array([
        [np.cos(theta),  0, np.sin(theta), 0],
        [      0,        1,       0,       0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [      0,        0,       0,       1]
    ])

def rot_z(vs, theta):
    theta = np.radians(theta)
    return vs @ np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [      0,              0,       1, 0],
        [      0,              0,       0, 1]
    ])

def move_x(vs, d):
    return vs @ np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [d, 0, 0, 1]
    ])

def move_y(vs, d):
    return vs @ np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, d, 0, 1]
    ])

def move_z(vs, d):
    return vs @ np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, d, 1]
    ])

def resize_x(vs, k):
    return vs @ np.array([
        [k, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def resize_y(vs, k):
    return vs @ np.array([
        [1, 0, 0, 0],
        [0, k, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def resize_z(vs, k):
    return vs @ np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, k, 0],
        [0, 0, 0, 1]
    ])

def mirror_x(vs):
    return vs @ np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

def mirror_y(vs):
    return vs @ np.array([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

def mirror_z(vs):
    return vs @ np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def lines_3d(basis, vs):
    vs = complete(vs)
    vs_coords = uncomplete(vs @ basis).T.ravel()

    n_coords = vs_coords.shape[0]

    vs_x, vs_y, vs_z = vs_coords[:int(n_coords/3)], vs_coords[int(n_coords/3):int(n_coords/3)*2], vs_coords[int(n_coords/3)*2:]

    lines = [
        [[vs_x[0], vs_x[1]], [vs_y[0], vs_y[1]], [vs_z[0], vs_z[1]]],
        [[vs_x[0], vs_x[2]], [vs_y[0], vs_y[2]], [vs_z[0], vs_z[2]]],
        [[vs_x[0], vs_x[4]], [vs_y[0], vs_y[4]], [vs_z[0], vs_z[4]]],

        [[vs_x[1], vs_x[3]], [vs_y[1], vs_y[3]], [vs_z[1], vs_z[3]]],
        [[vs_x[1], vs_x[5]], [vs_y[1], vs_y[5]], [vs_z[1], vs_z[5]]],

        [[vs_x[2], vs_x[3]], [vs_y[2], vs_y[3]], [vs_z[2], vs_z[3]]],
        [[vs_x[2], vs_x[6]], [vs_y[2], vs_y[6]], [vs_z[2], vs_z[6]]],

        [[vs_x[3], vs_x[7]], [vs_y[3], vs_y[7]], [vs_z[3], vs_z[7]]],


        [[vs_x[4], vs_x[6]], [vs_y[4], vs_y[6]], [vs_z[4], vs_z[6]]],
        [[vs_x[4], vs_x[5]], [vs_y[4], vs_y[5]], [vs_z[4], vs_z[5]]],

        [[vs_x[5], vs_x[7]], [vs_y[5], vs_y[7]], [vs_z[5], vs_z[7]]],

        [[vs_x[6], vs_x[7]], [vs_y[6], vs_y[7]], [vs_z[6], vs_z[7]]]
    ]

    return lines

def complete(vs):
    vs_copy = deepcopy(vs.tolist())
    for v in vs_copy:
        v.append(1)

    return np.array(vs_copy)

def uncomplete(vs):
    vs_copy = []
    for v in vs:
        vs_copy.append(deepcopy(v)[:-1])

    return np.array(vs_copy)

def scale(ax, s=1, a=0.):
    ax.plot([-s, s], [0, 0], [0, 0], alpha=a)
    ax.plot([0, 0], [-s, s], [0, 0], alpha=a)
    ax.plot([0, 0], [0, 0], [-s, s], alpha=a)

funcs = {
    'move': {
        'x': move_x,
        'y': move_y,
        'z': move_z
    },
    'resize': {
        'x': resize_x,
        'y': resize_y,
        'z': resize_z
    },
    'mirror': {
        'x': mirror_x,
        'y': mirror_y,
        'z': mirror_z
    },
    'rot': {
        'x': rot_x,
        'y': rot_y,
        'z': rot_z
    }
}

cube_color = 'black'

basis_w = deepcopy(basis)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(projection='3d')

def submit(text):
    global basis_w
    stdin = text
    print(stdin);

    if stdin.lower() == 'reset':
        basis_w = deepcopy(basis)
    elif len(stdin.split(' ')) == 2:
        assert stdin.split(' ')[0] == 'mirror', 'Unknown command'
        
        action, axis = stdin.lower().split(' ')
        basis_w = funcs[action][axis](basis_w)
    else:
        action, axis, value = stdin.lower().split(' ')
        basis_w = funcs[action][axis](basis_w, float(value))
    
    ax.cla()
    draw()
    plt.draw()


def draw():
    global ax
    scale(ax, 2, 0.5)
    for coord_x, coord_y, coord_z in lines_3d(basis_w, vs):
        ax.plot(coord_x, coord_y, coord_z, c=cube_color)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

draw()

plt.subplots_adjust(bottom=0.2)
axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
text_box = TextBox(axbox, 'Evaluate', initial='')
text_box.on_submit(submit)


plt.show()
