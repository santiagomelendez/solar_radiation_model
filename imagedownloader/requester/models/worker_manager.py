from Queue import PriorityQueue
from threading import Thread, Event
from datetime import datetime
import pytz
import traceback
import sys
import os
from splinter import Browser
from libs.console import total_seconds

try:
	from pyvirtualdisplay import Display
except ImportError:
	class Display(object):
		def start(self):
			pass
		def stop(self):
			pass

def print_exception():
	exc_type, exc_value, exc_traceback = sys.exc_info()
	lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
	print ''.join('!! ' + line for line in lines)


class Worker(object):
	def __init__(self, the_id, manager):
		self.id = the_id
		self._manager = manager
		self.running = False
		self.inform("initialized!")
	def inform(self, message):
		print "\033[32;1mWorker" + str(self.id) + " " + str(message) + "\033[0m"
	def start_working(self):
		self.running = True
		def functor(worker, manager):
			while worker.running:
				task = manager.get_job(self)
				if hasattr(task, 'run'):
					try:
						task.ready = False
						begin = datetime.utcnow().replace(tzinfo=pytz.UTC)
						task.run(manager)
						end = datetime.utcnow().replace(tzinfo=pytz.UTC)
						task.ready = True
						self.inform("finish in " + str(total_seconds(end - begin)) + " seconds (" + task.__class__.__name__ + ").")
					except Exception as e:
						self.inform("has captured an exception on (" + str(task.__class__.__name__)+ "): " + str(e))
						print_exception()
						manager.put_job(task)
			self.inform("shutdowned!")
		self._thread = Thread(target=functor, args=(self, self._manager,))
		self._thread.start()
	def stop_working(self):
		self.running = False
		self.inform("begining the shutdown!")

class BrowserManager(object):
	def __init__(self):
		self._lock = False
	def bootup(self):
		self._display = Display(visible=0, size=(1024, 768))
		self._display.start()
		profile = {}
		if 'HTTP_PROXY' in os.environ:
			proxy_url = os.environ['HTTP_PROXY']
			proxy_server = proxy_url.split(':')[1][2:]
			proxy_port = proxy_url.split(':')[-1]
			profile['network.proxy.type'] = 1
			profile['network.proxy.http'] = proxy_server
			profile['network.proxy.http_port'] = proxy_port
			profile['network.proxy.https'] = proxy_server
			profile['network.proxy.https_port'] = proxy_port
		self.browser = Browser(profile_preferences=profile)
	def obtain(self,background):
		while self._lock:
			background.wait('Browser lock', 15)
		self._lock = True
		return self.browser
	def release(self):
		self._lock = False
	def shutdown(self):
		self.browser.quit()
		self._display.stop()

class WorkerManager(object):
	class __impl(object):
		def __init__(self, worker_amount):
			self._queue = PriorityQueue()
			self._workers = [Worker(i,self) for i in range(worker_amount)]
			self._locked = False
			self._waiter = {}
			self.running = False
			self.browser_mgr = BrowserManager()
		def get_waiter(self, named):
			if not named in self._waiter:
				self._waiter[named] = Event()
			return self._waiter[named]
		def wait(self, named ,seconds):
			self.get_waiter(named).wait(seconds)
		def get_job(self, worker):
			while (self.running and (self._locked or self._queue.empty())):
				self.wait('get_job_for'+str(worker.id), 0.1)
			self._locked = True
			job = self._queue.get()
			self._queue.task_done()
			self._locked = False
			return job
		def put_job(self, job):
			self._queue.put(job)
		def start_workers(self):
			self.browser_mgr.bootup()
			self.running = True
			for worker in self._workers:
				worker.start_working()
		def stop_workers(self):
			self.running = False
			for worker in self._workers:
				worker.stop_working()
		def __del__(self):
			self.browser_mgr.shutdown()
	__instance = None
	def __init__(self, worker_amount):
		if WorkerManager.__instance is None:
			WorkerManager.__instance = WorkerManager.__impl(worker_amount)
		self.__dict__['_WorkerManager__instance'] = WorkerManager.__instance

	def __getattr__(self, attr):
		return getattr(self.__instance, attr)
	def __setattr__(self, attr, value):
		return setattr(self.__instance, attr, value)
