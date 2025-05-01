
import msgParser
import carState
import carControl
import threading
import neuralNet
import re

class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self, stage):
        '''Constructor'''
        self.WARM_UP = 0
        self.QUALIFYING = 1
        self.RACE = 2
        self.UNKNOWN = 3
        self.stage = stage

        self.nn = neuralNet.neuralNet(24, [16, 8], 3)
        
        self.parser = msgParser.MsgParser()
        
        self.state = carState.CarState()
        
        self.control = carControl.CarControl()
        
        self.steer_lock = 0.785398
        self.max_speed = 100
        self.prev_rpm = None

        # self.running = False
        # self.writer_thread = threading.Thread(target=self.csv_update)
        # self.writer_thread.daemon = True
        # self.writer_thread.start()
    
    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]
        
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        
        for i in range(5, 9):
            self.angles[i] = -20 + (i-5) * 5
            self.angles[18 - i] = 20 - (i-5) * 5
        
        return self.parser.stringify({'init': self.angles})
    
    def stop(self):
        """Stop the CSV writing thread"""
        self.running = False
        self.writer_thread.join()
    
    def csv_update(self): 
        self.running = True
        while self.running:
            if len(self.state.data_queue) > 0:
                data = self.state.data_queue.pop()
                self.state._write_to_csv(data)

    
    def drive(self, msg):
        self.state.setFromMsg(msg)

        self.nn.set_input_vector(self.state.angle, self.state.trackPos, self.state.speedX, self.state.rpm, self.state.gear, self.state.track)

        accel, steer, brake = self.nn.feed_forward()

        if accel > brake:
            brake = 0.0
        else:
            accel = 0.0

        self.control.setSteer(steer)
        self.control.setAccel(accel)
        self.control.setBrake(brake)
        self.gear()
        # self.control.setGear(self.decide_gear(gear))

        # print self.state.sensors 
        # self.steer()
        
        # self.gear()
        
        # self.speed()
        
        return self.control.toMsg()
        # return self.parser.stringify(self.state.sensors)
    
    def steer(self):
        angle = self.state.angle
        dist = self.state.trackPos
        
        self.control.setSteer((angle - dist*0.5)/self.steer_lock)
    
    def gear(self):
        rpm = self.state.getRpm()
        gear = self.state.getGear()
        
        if self.prev_rpm == None:
            up = True
        else:
            if (self.prev_rpm - rpm) < 0:
                up = True
            else:
                up = False
        
        if up and rpm > 7000:
            gear += 1
        
        if not up and rpm < 3000:
            gear -= 1
        
        self.control.setGear(gear)
    
    def speed(self):
        speed = self.state.getSpeedX()
        accel = self.control.getAccel()
        
        if speed < self.max_speed:
            accel += 0.1
            if accel > 1:
                accel = 1.0
        else:
            accel -= 0.1
            if accel < 0:
                accel = 0.0
        
        self.control.setAccel(accel)
            
        
    def onShutDown(self):
        self.stop()
        pass
    
    def onRestart(self):
        pass
        