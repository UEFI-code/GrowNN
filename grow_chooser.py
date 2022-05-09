import random
class Chooser():
    def __init__(self, r):
        self.level = 1.0
        self.r = r

    def decider(self):
        value = random.random()
        if value < self.level:
            self.level *= self.r
            print('GrowChooser Level Now %f' % self.level) 
            return True
        return False
