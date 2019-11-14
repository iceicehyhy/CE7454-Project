class Test():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def process(self):
        print (self.x + self.y)

if __name__  == '__main__':
    test = Test(1,2)
    test