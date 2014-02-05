# -*- coding: utf-8 -*- 
from plumbing.models import ComplexProcess, Stream, Material, MaterialStatus
from django.test import TestCase
from datetime import datetime
import pytz
import aspects


class TestComplexProcesses(TestCase):
	fixtures = [ 'initial_data.yaml', '*']

	def setUp(self):
		self.complex_process = ComplexProcess.objects.get_or_create(name='Filter nights and compact')[0]
		self.stream = Stream(root_path="/var/service/data/GVAR_IMG/argentina/")
		self.stream.save()
		self.material = Material()
		self.material.save()
		self.material_status = MaterialStatus(stream=self.stream, material=self.material)
		self.material_status.save()

	def test_encapsulate_in_array(self):
		# check if the __str__ method is defined to return the object root_path parameter.
		self.assertEquals(self.complex_process.encapsulate_in_array(self.stream), [self.stream])
		self.assertEquals(self.complex_process.encapsulate_in_array([self.stream]), [self.stream])

	def test_get_ordered_subprocsesses(self):
		# check if return the subprocesses ordered by ProcessOrder.position.
		ps = self.complex_process.get_ordered_subprocesses()
		previous = None
		actual = None
		for p in ps:
			if previous:
				self.assertTrue(previous.used_by_position >= p.used_by.position)

	def wrap_subprocess(self):
		# wrap all the 'do' mehtods of each subprocess, and register when a subprocesses is executed.
		def fake_get_ordered_subprocesses(*args):
			subprocesses = yield aspects.proceed(*args)
			dos = [ po.do for po in subprocesses ]
			def fake_do(*args):
				input_stream = args[1]
				self.called_subprocesses.append(args[0])
				# patch the results of any process to go through all the subprocesses.
				tmp_results = yield aspects.proceed(*args)
				results = self.complex_process.encapsulate_in_array(tmp_results)
				for r in results:
					result_materials = [ ms.material for ms in r.materials.all() ]
					for ms in input_stream.materials.all():
						if not ms.material in result_materials:
							ms.clone_for(r)
				yield aspects.return_stop(results)
			aspects.with_wrap(fake_do, *dos)
			yield aspects.return_stop(subprocesses)
		# check if execute each subprocess and collect all the results by wrapping the subprocesses do method.
		get_ordered_subprocesses = [ self.complex_process.get_ordered_subprocesses ]
		aspects.with_wrap(fake_get_ordered_subprocesses, *get_ordered_subprocesses)

	def test_do(self):
		# wrap all the 'do' mehtods of children process.
		self.called_subprocesses = []
		self.wrap_subprocess()
		resultant_stream = self.complex_process.do(self.stream)[0]
		# check if the amount of subprocesses called is consistent with the subprocess of the complex_process.
		self.assertEquals(len(self.called_subprocesses), self.complex_process.processes.count())
		# check if the order of each subprocess was right.
		for i in range(len(self.called_subprocesses)):
			self.assertEquals(self.called_subprocesses[i], self.complex_process.get_ordered_subprocesses()[i])
		# check if the result contain all the materials of the input stream.
		self.assertEquals(resultant_stream.materials.count(), self.stream.materials.count())