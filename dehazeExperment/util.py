import time
from datetime import datetime


def get_timestamp_now():
    '''
    获取当前的时间戳，为integer型
    :return: int型的时间戳
    '''
    thistime = int(time.mktime(datetime.now().timetuple()))
    return thistime
