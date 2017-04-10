class Particlelist:
#    def __init__(self, L):
#		# range of particle [0, L)		
#		self._L = L
    @property
    def x(self):
        return self._x
    @x.setter
    def x(self, value):
        self._x = value

    @property
    def p(self):
        return self._x
    @p.setter
    def p(self, value):
        self._p = value

    @property
    def id(self):
        return self._id
    @id.setter
    def id(self, value):
        if type(value) != int:
            raise ValueError("type must be integer")
        self._id = value

#    def move(self, dt):
#        self._x += (self._p*dt) % self._L
