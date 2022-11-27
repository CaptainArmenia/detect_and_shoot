from time import sleep

from detect_and_shoot import turn_on_pump, turn_off_pump, turn_on_lights, turn_off_lights, go_to_sleep

print("Turning off pump and lights")
sleep(0.1)
turn_off_pump()
sleep(0.1)
turn_off_lights()