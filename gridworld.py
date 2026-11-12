class GridWorld:
    def __init__(self, size=5, goal=None, walls=None):
        self.size = size
        self.start = (0, 0)
        if goal is None:
            self.goal = (size - 1, size - 1)
        else:
            self.goal = goal
        if walls is None:
            self.walls = set()
        else:
            self.walls = set(walls)
        self.agent = self.start

    def reset(self):
        self.agent = self.start
        return self.agent

    def step(self, action):
        # Actions: 0 = up, 1 = down, 2 = left, 3 = right
        row, col = self.agent
        new_row, new_col = row, col

        if action == 0:
            new_row = row - 1
        elif action == 1:
            new_row = row + 1
        elif action == 2:
            new_col = col - 1
        elif action == 3:
            new_col = col + 1

        # Stay inside the grid
        new_row = max(0, min(self.size - 1, new_row))
        new_col = max(0, min(self.size - 1, new_col))

        # Walls block movement
        if (new_row, new_col) in self.walls:
            new_row, new_col = row, col

        self.agent = (new_row, new_col)
        done = (self.agent == self.goal)

        # Reward: +10 at the goal, -1 for every other step
        if done:
            reward = 10
        else:
            reward = -1

        return self.agent, reward, done

    def show(self):
        for r in range(self.size):
            line = ""
            for c in range(self.size):
                if (r, c) == self.agent:
                    line += "A "
                elif (r, c) == self.goal:
                    line += "G "
                elif (r, c) in self.walls:
                    line += "# "
                else:
                    line += ". "
            print(line)
        print()