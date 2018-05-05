import time
import math

def asMinutes(s) :
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent) :
    now = time.time()
    s = now-since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def printTime(start_time) :
    print("%.0f minutes passed" % ((float(time.time()) - float(start_time))/60) )