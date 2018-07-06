from time import sleep
from typing import Union

####
# user can do sleep(0.01) or sleep(1)
####
def sleep_sec(seconds: Union[float, int]):
    sleep(seconds) 
