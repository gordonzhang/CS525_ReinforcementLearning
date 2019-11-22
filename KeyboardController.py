from Environment import Environment
from Utils import Agent, Direction, Position, StepResponse

import curses
import time


env = Environment(width=5, height=5, num_agents=1, start=Position(0,0), goal=Position(4,4), view_range=2, render=True)

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
            result = agents[0]

            state = result.state
            reward = result.reward
            done = result.done

            if done:
                pStr = None
            else:
                pStr = "\t%s\n\t%s\n\t%s\n\t%s" % ( \
                    list(state.observation.up), \
                    list(state.observation.right), \
                    list(state.observation.down), \
                    list(state.observation.left) )

            stdscr.addstr("Reward: " + str(reward))
            stdscr.addstr("\nAm I done? " + str(done))
            stdscr.addstr("\nI see: " + str(pStr))
            
            stdscr.refresh()
            stdscr.move(0, 0)
            time.sleep(.01)

    time.sleep(2)


if __name__ == '__main__':
    curses.wrapper(main)