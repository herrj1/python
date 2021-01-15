class hashtable():
    def __init__(self):
        self.size = 1000
        self.table = [None] * self.size

    def add(self,item):
        hashcode = self.hash(item)
        if hashcode >= self.size:
            self.size *= 2
        else:
            self.table[hashcode] = item


    def hash(self, key):
        return key % size


