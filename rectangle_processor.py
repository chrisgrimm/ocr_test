import numpy as np
import cv2

class RectangleProcessor(object):

    def __init__(self, text_image_original_size, mask_image):
        puddles = self.compute_rectangles(mask_image)
        for i, puddle in enumerate(puddles.values()):
            puddle_image = self.extract_puddle_image(text_image_original_size, puddle, mask_image.shape[1], mask_image.shape[0])
            cv2.imwrite('puddle_dir/puddle_%s.png' % i, puddle_image)



    def compute_rectangles(self, mask_image):
        threshold_cutoff = 0.9
        mask_image = mask_image[:, :, 0] / 255.
        mask_image[mask_image < threshold_cutoff] = 0.0
        mask_image[mask_image > 0] = 1.0
        # get all unvisited, white pixels.
        unvisited_pixels = set([(x, y) for x in range(mask_image.shape[1])
                                for y in range(mask_image.shape[0])
                                if mask_image[y, x] == 1.0])
        puddles = dict()
        width, height = mask_image.shape[1], mask_image.shape[0]
        while unvisited_pixels:
            current_pixel = unvisited_pixels.pop()
            puddle = self.flood_fill(current_pixel, unvisited_pixels, width, height)
            puddles[current_pixel] = puddle
        puddles = {key: self.touch_up_puddle(puddle, width, height) for key, puddle in puddles.items()}
        return puddles




    def get_neighbors(self, xy, width, height):
        (x, y) = xy
        neighbors = [(x, y+1), (x, y-1), (x+1, y), (x-1, y)]
        neighbors = [(np.clip(x, 0, width-1), np.clip(y, 0, height-1))
                        for (x, y) in neighbors]
        neighbors = list(set(neighbors))
        return neighbors


    def flood_fill(self, xy, unvisited, width, height):
        puddle = set()
        fringe = set([xy])
        while fringe:
            (x, y) = fringe.pop()
            if (x, y) in unvisited:
                unvisited.remove((x,y))
            puddle.add((x,y))
            neighbors = [xy for xy in self.get_neighbors((x,y), width, height)
                         if xy in unvisited]
            for neighbor in neighbors:
                fringe.add(neighbor)
        return puddle

    def touch_up_puddle(self, puddle, width, height):
        for x in range(width):
            for y in range(height):
                if (x,y) in puddle:
                    continue
                neighbors = self.get_neighbors((x,y), width, height)
                if len(set(neighbors).intersection(puddle)) >= 3:
                    puddle.add((x,y))
        return puddle





    def extract_puddle_image(self, text_image_original, puddle, width, height):
        canvas = np.zeros((height, width))
        for (x,y) in puddle:
            canvas[y, x] = 1.0
        #return 255*np.tile(np.reshape(canvas, [height, width, 1]), [1, 1, 3])
        #text_image_original = cv2.resize(text_image_original, (128, 128))
        h_orig, w_orig = text_image_original.shape[0], text_image_original.shape[1]
        canvas = cv2.resize(canvas, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        #cv2.imshow('puddle', canvas)d
        #cv2.waitKey()
        canvas_pixels = set([(x, y) for x in range(canvas.shape[1])
                            for y in range(canvas.shape[0])
                            if canvas[y, x] == 1.0])
        max_x = max(canvas_pixels, key=lambda xy: xy[0])[0]
        min_x = min(canvas_pixels, key=lambda xy: xy[0])[0]
        max_y = max(canvas_pixels, key=lambda xy: xy[1])[1]
        min_y = min(canvas_pixels, key=lambda xy: xy[1])[1]
        copied_original = np.copy(text_image_original)
        for x in range(w_orig):
            for y in range(h_orig):
                if (x,y) not in canvas_pixels:
                    copied_original[y,x,:] = 0
        cropped_original = copied_original[min_y:max_y, min_x:max_x, :]
        print('boop!')
        return cropped_original







if __name__ == '__main__':
    original_image = cv2.imread('./test_images/6_text.jpg')
    mask_image = cv2.imread('./result_dir/0_mask.png')
    RectangleProcessor(original_image, mask_image)



