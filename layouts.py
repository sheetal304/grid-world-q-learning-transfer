from gridworld import GridWorld

# Walls used by the source layout A (several target layouts reuse them)
WALLS_A = [(1, 1), (2, 2), (3, 3)]


def make_layout_a():
    # Layout A (source): diagonal walls, goal at (4, 4)
    return GridWorld(walls=WALLS_A)


# ---- Target layouts where ONLY THE WALLS change (goal stays at (4, 4)) ----

def make_layout_b():
    # Layout B: original wall-change layout
    return GridWorld(walls=[(0, 2), (1, 2), (3, 1), (3, 2), (3, 3)])


def make_layout_d():
    # Layout D: two long walls that force a zig-zag route
    return GridWorld(walls=[(1, 0), (1, 1), (1, 2), (3, 2), (3, 3), (3, 4)])


def make_layout_e():
    # Layout E: a wall across the middle row with one gap at (2, 2)
    return GridWorld(walls=[(2, 0), (2, 1), (2, 3), (2, 4)])


# ---- Target layouts where ONLY THE GOAL changes (walls same as A) ----

def make_layout_c():
    # Layout C: goal moved to the top-right corner
    return GridWorld(goal=(0, 4), walls=WALLS_A)


def make_layout_f():
    # Layout F: goal moved to the bottom-left corner
    return GridWorld(goal=(4, 0), walls=WALLS_A)


def make_layout_g():
    # Layout G: goal moved to the right edge, middle row
    return GridWorld(goal=(2, 4), walls=WALLS_A)


# name, what changes, function
TARGETS = [
    ("B", "walls change", make_layout_b),
    ("D", "walls change", make_layout_d),
    ("E", "walls change", make_layout_e),
    ("C", "goal changes", make_layout_c),
    ("F", "goal changes", make_layout_f),
    ("G", "goal changes", make_layout_g),
]