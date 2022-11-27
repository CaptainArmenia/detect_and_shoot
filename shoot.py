from time import sleep
import argparse

from detect_and_shoot import turn_on_pump, turn_off_pump


parser = argparse.ArgumentParser()
parser.add_argument("--time", type=int, default=1)
args = parser.parse_args()

print(f"Shooting for {args.time} s")
turn_on_pump()
sleep(args.time)
turn_off_pump()

