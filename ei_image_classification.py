# Edge Impulse - OpenMV Image Classification Example

import sensor, image, time, os, tf, pyb
from pyb import Pin, Timer


LampControlPin = Pin("P0", Pin.OUT_PP)
TvControlPin = Pin("P1", Pin.OUT_PP)
LampTimer = Timer(4)
TvTimer = Timer(2)


sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.skip_frames(time=2000)          # Let the camera adjust.

net = "trained.tflite"
labels = [line.rstrip('\n') for line in open("labels.txt")]
LampCounter = 0
TvCounter = 0

def increment_lamp_counter():
    global LampCounter
    LampCounter += 1
    return (LampCounter > 5)

def reset_lamp_counter():
    global LampCounter
    LampCounter = 0

def increment_tv_counter():
    global LampCounter
    LampCounter += 1
    return (LampCounter > 5)

def reset_tv_counter():
    global LampCounter
    LampCounter = 0

def deactivateLampPin(timer):
    print("inside calllback")
    if increment_lamp_counter():
        LampControlPin.low()
        timer.deinit()
        reset_lamp_counter()

def deactivateTvPin(timer):
    if increment_tv_counter():
        TvControlPin.low()
        timer.deinit()
        reset_tv_counter()

def activateLampPin():
    print("activating pins")
    LampTimer.deinit()
    LampControlPin.high()
    LampTimer.init(freq = 1)
    LampTimer.callback(deactivateLampPin)

def activateTelevisionPin():
    print("activating pins")
    TvTimer.deinit()
    TvControlPin.high()
    TvTimer.init(freq = 1)
    TvTimer.callback(deactivateTvPin)


clock = time.clock()
LampControlPin.low()
TvControlPin.low()


while(True):
    clock.tick()

    img = sensor.snapshot()

    # default settings just do one detection... change them to search the image...
    for obj in tf.classify(net, img, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
#        print("**********\nPredictions at [x=%d,y=%d,w=%d,h=%d]" % obj.rect())
        img.draw_rectangle(obj.rect())
        # This combines the labels and confidence values into a list of tuples
        predictions_list = list(zip(labels, obj.output()))

#        for i in range(len(predictions_list)):
#            print("%s = %f" % (predictions_list[i][0], predictions_list[i][1]))

        if predictions_list[0][1] > 0.6:
            activateLampPin()


        elif predictions_list[1][1] > 0.6:
            activateTelevisionPin()



    print("lampPin: %d" % LampControlPin.value())
    print("TV Pin: %d" % TvControlPin.value())


    print(clock.fps(), "fps")

