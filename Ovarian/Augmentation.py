import Augmentor
import torchvision

p = Augmentor.Pipeline("C:\dataset\Validation\Invalid")


p.flip_left_right(probability=1)

p.process()
