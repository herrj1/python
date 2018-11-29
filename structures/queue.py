#Python queue ADT

class Queue:
	def __init__(self):
		self.items = []

	def is_empty(self):
		return self.items == []

	def enqueue(self, item):
		self.items.insert(0,item)

	def dequeue(self):
		return self.items.pop()

	def size(self):
		return len(self.items)


#Sample use
q = Queue()
q.enqueue('salutation')
q.enqueue('Cat')
q.enqueue(22)
q.dequeue(10)
q.size()