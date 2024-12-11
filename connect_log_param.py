import logging
import time
import os

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.crazyflie.log import LogConfig
from cflib.positioning.motion_commander import MotionCommander

home_directory = os.path.expanduser("~")  
log_file_path = os.path.join(home_directory, 'logging.txt') 

logging.basicConfig(
    filename=log_file_path,            # File in the home directory
    filemode='w',                      # Overwrite the file each run
    level=logging.INFO,                # Log level
    format='%(asctime)s - %(message)s' # Log format
)
 
uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

import time

def take_off_simple(scf):
    """
    Controls the Crazyflie to take off using thrust commands directly.
    """
    print("Taking off!")

    cf = scf.cf

    thrust = 10
    roll = 0        
    pitch = 0       
    yaw_rate = 0    

    with MotionCommander(scf) as mc:
        #mc.take_off(None, 0.0001)
        #time.sleep(1)
        mc.take_off(None, -0.0001)
        time.sleep(1)
        #mc.forward(None, 10)

        time.sleep(3)
        mc.stop()

    for _ in range(50):  
        cf.commander.send_setpoint(roll, pitch, yaw_rate, thrust)
        time.sleep(0.02) 

    thrust = 35000  
    for _ in range(100):  
        cf.commander.send_setpoint(roll, pitch, yaw_rate, thrust)
        time.sleep(0.02)

    cf.commander.send_setpoint(0, 0, 0, 0)
    print("Landed!")



def log_stab_callback(timestamp, data, logconf):
    """Logs roll, pitch, and yaw values."""
    roll = data.get('stabilizer.roll', 0)
    pitch = data.get('stabilizer.pitch', 0)
    yaw = data.get('stabilizer.yaw', 0)

    logging.info(f'Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}')


def simple_log_async(scf):
    """Sets up logging for roll, pitch, and yaw."""
    print("Logging...")
    cf = scf.cf
    log_conf = LogConfig(name='Stabilizer', period_in_ms=10)
    log_conf.add_variable('stabilizer.roll', 'float')
    log_conf.add_variable('stabilizer.pitch', 'float')
    log_conf.add_variable('stabilizer.yaw', 'float')

    cf.log.add_config(log_conf)
    log_conf.data_received_cb.add_callback(log_stab_callback)
    log_conf.start()

    time.sleep(10)
    print("Logged.")
    log_conf.stop()


if __name__ == '__main__':
    cflib.crtp.init_drivers()
    print("Trying to connect...")
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        print("Connected.")
        time.sleep(5)
        take_off_simple(scf)
        simple_log_async(scf)

