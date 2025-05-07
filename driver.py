
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
    def getNueralNetwork(self):
        return self.nn
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
        if hasattr(self, 'writer_thread') and self.writer_thread is not None:
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

        # Steering control
        self.control.setSteer(steer)
        # Apply our controlled acceleration and braking
        self.control.setAccel(accel)
        self.control.setBrake(brake)
        # Handle gear changes
        self.gear()
        return self.control.toMsg(), [accel, steer, brake]
    
    def steer(self):
        angle = self.state.angle
        dist = self.state.trackPos
        
        self.control.setSteer((angle - dist*0.5)/self.steer_lock)
        return (angle - dist*0.5)/self.steer_lock
    def getsteer(self):
        angle = self.state.angle
        dist = self.state.trackPos
        return (angle - dist*0.5)/self.steer_lock

    def gear(self):
        rpm = self.state.getRpm()
        speed = self.state.getSpeedX()
        gear = self.state.getGear()
        MIN_MOVING_SPEED = 5  # km/h
        GEAR_CHANGE_RPM_BUFFER = 500  # RPM buffer to prevent rapid shifting

        # Initialize previous RPM if None
        if self.prev_rpm is None:
            self.prev_rpm = rpm

        # Only consider gear changes if car is actually moving
        if speed > MIN_MOVING_SPEED:
            # Determine RPM trend (rising/falling)
            rpm_rising = rpm > self.prev_rpm

            # Upshift logic (only if RPM is rising and in safe range)
            if rpm > (7000 - GEAR_CHANGE_RPM_BUFFER) and rpm_rising:
                if gear < 6:  # Prevent exceeding max gear
                    gear += 1

            # Downshift logic (only if RPM is falling and in safe range)
            elif rpm < (3000 + GEAR_CHANGE_RPM_BUFFER) and not rpm_rising:
                if gear > 1:  # Prevent going below min gear
                    gear -= 1

        # Emergency stop detection (crash or spin)
        if speed < MIN_MOVING_SPEED:
            # Reset to first gear when stopped or damaged
            gear = 1

        # Ensure gear stays in valid range
        gear = max(1, min(6, gear))
        
        self.control.setGear(gear)
        self.prev_rpm = rpm  # Update RPM history
    
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


    def __del__(self):
        pass
        