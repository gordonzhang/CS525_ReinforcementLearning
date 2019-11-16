from Environment import Environment

import curses
import time


env = Environment(width=5, height=5, num_agents=1, start=(0,0), goal=(4,4), view_range=2, render=True)

def main(stdscr):
    # do not wait for input when calling getch
    stdscr.nodelay(1)

    done = False

    while not done:
        # get keyboard input, returns -1 if none available
        c = stdscr.getch()
        if c >= 258 and c <= 261:
            # print numeric value
            if c == 258:
                a = 2
            elif c == 259:
                a = 0
            elif c == 260:
                a = 3
            elif c == 261:
                a = 1
            
            agents = env.step({0:a})
            state, reward, done, _ = agents[0]
            stdscr.addstr("Reward: " + str(reward))
            # print("Reward:", reward)
            stdscr.refresh()
            stdscr.move(0, 0)
            time.sleep(.01)

    time.sleep(1)


if __name__ == '__main__':
    curses.wrapper(main)