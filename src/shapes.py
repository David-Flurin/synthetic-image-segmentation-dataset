from matplotlib.colors import to_rgb
import numpy as np
from numpy_turtle import Turtle as np_turtle
import math
import scipy
from skimage.segmentation import flood_fill
from .utils import rotate

class Shape():

    FOOTPRINT = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    def __init__(self, radius, pos, img_size):
        self.radius = radius
        self.pos = pos
        self.img_size = img_size
        assert(self.pos[0] < self.img_size[0] and self.pos[1] < self.img_size[1])


    def fill(self):
        last_border = self.img_size[0] * [False]
        inside = self.img_size[0] * [-1]
        for x in range(self.img_size[0]):
            for y in range(self.img_size[1]):
                if self.mask[x,y] < 0.5:
                    if inside[y] == -1 and last_border[y]:
                        inside[y] = x
                    else:
                        continue
                else:
                    if inside[y] == -1:
                        last_border[y] = 1
                        continue
                    if inside[y] >= 0:
                        self.mask[inside[y]:x, y] = np.ones(x-inside[y])
                        inside[y] = -1
                        last_border[y] = True
        

    def draw(self):
        #img = np.repeat(self.mask[:, :, np.newaxis], 3, axis=2)*255

        from matplotlib import pyplot as plt
        plt.imshow(self.mask, interpolation='nearest')
        plt.show()

class Circle(Shape):
    def __init__(self, radius, pos, img_size, rotation):
        super().__init__(radius, pos, img_size)
        self.seg_color = np.array(to_rgb('navy'))
        xx, yy = np.mgrid[:img_size[0], :img_size[1]]
        distance = (xx - pos[0])**2 + (yy - pos[1])**2
        thresh = radius**2
        self.mask = np.where(distance <= thresh, 1, 0)


class Triangle(Shape):
    def __init__(self, radius, center, img_size, rotation):
        super().__init__(radius, center, img_size)
        self.seg_color = np.array(to_rgb('olive'))
        start = (center[0]-radius, center[1])
        self.coordinates = rotate(center, start, math.radians(rotation))
        self.rotation = rotation
        self.generate()

    
    def generate(self):
        canvas = np.zeros((self.img_size[0], self.img_size[1], 4))
        t = np_turtle(canvas)
        t.position = self.coordinates[0], self.coordinates[1]
        t.rotate(-math.pi / 6 + math.radians(self.rotation))
        t.color = (0, 0, 0, 1)
        l = 2*self.radius
        a = 2 * math.pi / 3
        
        t.forward(l)
        t.rotate(a)
        t.forward(l)
        t.rotate(a)
        t.forward(l)
        
        self.mask = canvas[:, :, 3]
        flood_fill(self.mask, self.pos, 1., footprint=self.FOOTPRINT, in_place=True)

class Square(Shape):
    def __init__(self, radius, center, img_size, rotation):
        super().__init__(radius, center, img_size)
        self.seg_color = np.array(to_rgb('darkslategray'))
        start = (center[0]-radius, center[1]-radius)
        self.coordinates = rotate(center, start, math.radians(rotation))
        self.rotation = rotation
        self.generate()

    
    def generate(self):
        canvas = np.zeros((self.img_size[0], self.img_size[1], 4))
        t = np_turtle(canvas)
        t.position = self.coordinates[0], self.coordinates[1]
        t.rotate(math.radians(self.rotation))
        t.color = (0, 0, 0, 1)
        l = 2*self.radius
        a = math.pi / 2
        
        t.forward(l)
        t.rotate(a)
        t.forward(l)
        t.rotate(a)
        t.forward(l)
        t.rotate(a)
        t.forward(l)
        
        self.mask = canvas[:, :, 3]
        flood_fill(self.mask, self.pos, 1., footprint=self.FOOTPRINT, in_place=True)

class Pentagon(Shape):
    def __init__(self, radius, center, img_size, rotation):
        super().__init__(radius, center, img_size)
        self.seg_color = np.array(to_rgb('firebrick'))
        start = (center[0]-radius, center[1])
        self.coordinates = rotate(center, start, math.radians(rotation))
        self.rotation = rotation
        self.generate()

    
    def generate(self):
        canvas = np.zeros((self.img_size[0], self.img_size[1], 4))
        t = np_turtle(canvas)
        t.position = self.coordinates[0], self.coordinates[1]
        t.rotate(math.radians(self.rotation - 54))
        t.color = (0, 0, 0, 1)
        l = (4*self.radius) / math.sqrt(5 + 2 * math.sqrt(5))
        a = math.radians(72)
        
        t.forward(l)
        t.rotate(a)
        t.forward(l)
        t.rotate(a)
        t.forward(l)
        t.rotate(a)
        t.forward(l)
        t.rotate(a)
        t.forward(l)
        
        self.mask = canvas[:, :, 3]
        flood_fill(self.mask, self.pos, 1., footprint=self.FOOTPRINT, in_place=True)


class Hexagon(Shape):
    def __init__(self, radius, center, img_size, rotation):
        super().__init__(radius, center, img_size)
        self.seg_color = np.array(to_rgb('dimgray'))
        start = (center[0]-radius, center[1])
        self.coordinates = rotate(center, start, math.radians(rotation))
        self.rotation = rotation
        self.generate()

    
    def generate(self):
        canvas = np.zeros((self.img_size[0], self.img_size[1], 4))
        t = np_turtle(canvas)
        t.position = self.coordinates[0], self.coordinates[1]
        t.rotate(math.radians(self.rotation - 60))
        t.color = (0, 0, 0, 1)
        l = (2*self.radius) / math.sqrt(3)
        a = math.radians(60)
        
        t.forward(l)
        t.rotate(a)
        t.forward(l)
        t.rotate(a)
        t.forward(l)
        t.rotate(a)
        t.forward(l)
        t.rotate(a)
        t.forward(l)
        t.rotate(a)
        t.forward(l)
        
        self.mask = canvas[:, :, 3]
        flood_fill(self.mask, self.pos, 1., footprint=self.FOOTPRINT, in_place=True)


class Octagon(Shape):
    def __init__(self, radius, center, img_size, rotation):
        super().__init__(radius, center, img_size)
        self.seg_color = np.array(to_rgb('saddlebrown'))
        start = (center[0]-radius, center[1])
        self.coordinates = rotate(center, start, math.radians(rotation))
        self.rotation = rotation
        self.generate()

    
    def generate(self):
        canvas = np.zeros((self.img_size[0], self.img_size[1], 4))
        t = np_turtle(canvas)
        t.position = self.coordinates[0], self.coordinates[1]
        t.rotate(math.radians(self.rotation - 67.5))
        t.color = (0, 0, 0, 1)
        l = (2*self.radius) / (1 + math.sqrt(2))
        a = math.radians(45)
        
        t.forward(l)
        t.rotate(a)
        t.forward(l)
        t.rotate(a)
        t.forward(l)
        t.rotate(a)
        t.forward(l)
        t.rotate(a)
        t.forward(l)
        t.rotate(a)
        t.forward(l)
        t.rotate(a)
        t.forward(l)
        t.rotate(a)
        t.forward(l)
        
        self.mask = canvas[:, :, 3]
        flood_fill(self.mask, self.pos, 1., footprint=self.FOOTPRINT, in_place=True)


class Star(Shape):
    def __init__(self, radius, center, img_size, rotation):
        super().__init__(radius, center, img_size)
        self.seg_color = np.array(to_rgb('mediumvioletred'))
        offset = radius * ((3 - 2*math.sqrt(2)) + (3*math.sqrt(2) - 4)) 
        start = (center[0]-radius, center[1] - offset)
        self.coordinates = rotate(center, start, math.radians(rotation))
        self.rotation = rotation
        self.generate()

    
    def generate(self):
        canvas = np.zeros((self.img_size[0], self.img_size[1], 4))
        t = np_turtle(canvas)
        t.position = self.coordinates[0], self.coordinates[1]
        t.rotate(math.radians(self.rotation))
        t.color = (0, 0, 0, 1)
        l = self.radius * (2 - math.sqrt(2))
        a_l = 3*math.pi / 2
        a_s = 3*math.pi /4
        for i in range(8):
            t.forward(l)
            t.rotate(a_l)
            t.forward(l)
            t.rotate(a_s)
        t.forward(l)
        
        self.mask = canvas[:, :, 3]
        flood_fill(self.mask, self.pos, 1., footprint=self.FOOTPRINT, in_place=True)



class Cross(Shape):
    def __init__(self, radius, center, img_size, rotation):
        super().__init__(radius, center, img_size)
        self.seg_color = np.array(to_rgb('darkgoldenrod'))
        start = (center[0]-radius, center[1] - radius/3)
        self.coordinates = rotate(center, start, math.radians(rotation))
        self.rotation = rotation
        self.generate()

    
    def generate(self):
        canvas = np.zeros((self.img_size[0], self.img_size[1], 4))
        t = np_turtle(canvas)
        t.position = self.coordinates[0], self.coordinates[1]
        t.rotate(math.radians(self.rotation))
        t.color = (0, 0, 0, 1)
        l = self.radius / 1.5
        a_l = 3*math.pi / 2
        a_s = math.pi /2
        
        for i in range(4):
            t.forward(l)
            t.rotate(a_l)
            t.forward(l)
            t.rotate(a_s)
            t.forward(l)
            t.rotate(a_s)

        self.mask = canvas[:, :, 3]        
        flood_fill(self.mask, self.pos, 1., footprint=self.FOOTPRINT, in_place=True)
            
class Heart(Shape):
    def __init__(self, radius, pos, img_size, rotation):
        super().__init__(radius, pos, img_size)
        self.seg_color = np.array(to_rgb('darkgreen'))
        self.mask = np.zeros(img_size, dtype=np.float32)
        xx, yy = np.mgrid[:img_size[0], :img_size[1]]
        heartline = -((yy - img_size[0]//2)**2 + (xx - img_size[1]//2)**2 - radius**2)**3 - ((yy - img_size[0]//2)**2 * (xx - img_size[1]//2)**3) * 2*radius
        self.mask[np.where(heartline > 1)] = 1.
        self.mask = scipy.ndimage.rotate(self.mask, rotation, reshape=False)
        self.mask[np.where(self.mask < 0.9)] = 0.
        cropped = crop(self.mask)
        self.mask = np.zeros(img_size)
        cropped_origin = (pos[0]-cropped.shape[0]//2, pos[1]-cropped.shape[1]//2)
        self.mask[max(0,cropped_origin[0]):min(cropped_origin[0] + cropped.shape[0], img_size[0]), max(cropped_origin[1],0):min(cropped_origin[1] + cropped.shape[1], img_size[1])] = cropped[max(0, -cropped_origin[0]):min(cropped.shape[0], img_size[0]-cropped_origin[0]), max(0, -cropped_origin[1]):min(cropped.shape[1], img_size[1]-cropped_origin[1])]


def crop(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return img[rmin:rmax+1, cmin:cmax+1]






