from gpiozero import AngularServo
from time import sleep

servo = AngularServo(17, min_angle=-90, max_angle=90)

print("Full speed clockwise")
servo.angle = -90
sleep(2)

print("Stop")
servo.angle = 0
sleep(2)

print("Full speed counter-clockwise")
servo.angle = 90
sleep(2)

print("Slow crawl")
servo.angle = 10
sleep(2)

servo.angle = None
