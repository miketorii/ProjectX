import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

GPIO.setup(18, GPIO.OUT)

for x in range(10):
    GPIO.output(18, True)
    print("GPIO OUT True")
    time.sleep(2)
    GPIO.output(18, False)
    print("GPIO OUT False")    
    time.sleep(2)

GPIO.cleanup()

