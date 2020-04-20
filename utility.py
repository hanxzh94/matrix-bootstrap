global __indent
global __active
__indent = 0
__active = True

from datetime import datetime
def current_time():
	now = datetime.now()
	return now.strftime("%H:%M:%S.%f")

def disable_debug():
	global __active
	__active = False

def enable_debug():
	global __active
	__active = True

def debug(s):
	if __active:
		global __indent
		print("DEBUG %s: %s%s" % (current_time(), (" " * __indent), s))

def debug_start(s):
	if __active:
		global __indent
		print("DEBUG %s: %s%s" % (current_time(), (" " * __indent), s))
		__indent += 4

def debug_end(s):
	if __active:
		global __indent
		__indent -= 4
		print("DEBUG %s: %s%s" % (current_time(), (" " * __indent), s))

def expect(b):
	if __active:
		assert isinstance(b, bool) and b, "unexpected: %s" % b
