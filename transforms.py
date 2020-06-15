import torchvision.transforms as transforms

from PIL import Image

img = Image.open('tina.jpg')

'''
# CenterCrop
size = (224, 224)
transform = transforms.CenterCrop(size)
center_crop = transform(img)
center_crop.save('center_crop.jpg')

# ColorJitter
brightness = (1, 10)
contrast = (1, 10)
saturation = (1, 10)
hue = (0.2, 0.4)
transform = transforms.ColorJitter(brightness, contrast, saturation, hue)
color_jitter = transform(img)
color_jitter.save('color_jitter.jpg')

# FiveCrop
size = (224, 224)
transform = transforms.FiveCrop(size)
five_crop = transform(img)
for index, img in enumerate(five_crop):
    img.save(str(index) + '.jpg')

# Grayscale
transform = transforms.Grayscale()
grayscale = transform(img)
grayscale.save('grayscale.jpg')

# Compose, Pad
size = (224, 224)
padding = 16
fill = (0, 0, 255)
transform = transforms.Compose([
        transforms.CenterCrop(size),
        transforms.Pad(padding, fill)
])
pad = transform(img)
pad.save('pad.jpg')

# RandomAffine
degrees = (15, 30)
translate=(0, 0.2)
scale=(0.8, 1)
fillcolor = (0, 0, 255)
transform = transforms.RandomAffine(degrees=degrees, translate=translate, scale=scale, fillcolor=fillcolor)
random_affine = transform(img)
random_affine.save('random_affine.jpg')

# RandomApply
size = (224, 224)
padding = 16
fill = (0, 0, 255)
transform = transforms.RandomApply([transforms.CenterCrop(size), transforms.Pad(padding, fill)])
for i in range(3):
    random_apply = transform(img)
    random_apply.save(str(i) + '.jpg')

# RandomChoice
transform = transforms.RandomChoice([transforms.RandomAffine(degrees), 
                                     transforms.CenterCrop(size), 
                                     transforms.Pad(padding, fill)])
for i in range(3):
    random_order = transform(img)
    random_order.save(str(i) + '.jpg')

# RandomCrop
size = (224, 224)
transform = transforms.RandomCrop(size)
random_crop = transform(img)
random_crop.save('p.jpg')

# RandomGrayscale
p = 0.5
transform = transforms.RandomGrayscale(p)
for i in range(3):
    random_grayscale = transform(img)
    random_grayscale.save(str(i) + '.jpg')

# RandomHorizontalFlip
p = 0.5
transform = transforms.RandomHorizontalFlip(p)
for i in range(3):
    random_horizontal_filp = transform(img)
    random_horizontal_filp.save(str(i) + '.jpg')

# RandomOrder
size = (224, 224)
padding = 16
fill = (0, 0, 255)
degrees = (15, 30)
transform = transforms.RandomOrder([transforms.RandomAffine(degrees), 
                                    transforms.CenterCrop(size), 
                                    transforms.Pad(padding, fill)])
for i in range(3):
    random_order = transform(img)
    random_order.save(str(i) + '.jpg')

# RandomPerspective
distortion_scale = 1
p = 1
fill = (0, 0, 255)
transform = transforms.RandomPerspective(distortion_scale=distortion_scale, p=p, fill=fill)
random_perspective = transform(img)
random_perspective.save('random_perspective.jpg')

# RandomResizedCrop
size = (256, 256)
scale=(0.8, 1.0)
ratio=(0.75, 1.0)
transform = transforms.RandomResizedCrop(size=size, scale=scale, ratio=ratio)
random_resized_crop = transform(img)
random_resized_crop.save('random_resized_crop.jpg')

# RandomRotation
degrees = (15, 30)
fill = (0, 0, 255)
transform = transforms.RandomRotation(degrees=degrees, fill=fill)
random_rotation = transform(img)
random_rotation.save('random_rotation.jpg')

# RandomVerticalFlip
p = 1
transform = transforms.RandomVerticalFlip(p)
random_vertical_filp = transform(img)
random_vertical_filp.save('random_vertical_filp.jpg')

# Resize
size = (224, 224)
transform = transforms.Resize(size)
resize_img = transform(img)
resize_img.save('resize_img.jpg')

# ToPILImage
img = Image.open('tina.jpg')
transform = transforms.ToTensor()
img = transform(img)
print(img.size())
img_r = img[0, :, :]
img_g = img[1, :, :]
img_b = img[2, :, :]
print(type(img_r))
print(img_r.size())
transform = transforms.ToPILImage()
img_r = transform(img_r)
img_g = transform(img_g)
img_b = transform(img_b)
print(type(img_r))
img_r.save('img_r.jpg')
img_g.save('img_g.jpg')
img_b.save('img_b.jpg')

# ToTensor
img = Image.open('tina.jpg')
print(type(img))
print(img.size)
transform = transforms.ToTensor()
img = transform(img)
print(type(img))
print(img.size())
'''

# RandomErasing
p = 1.0
scale = (0.2, 0.3)
ratio = (0.5, 1.0)
value = (0, 0, 255)

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomErasing(p=p, scale=scale, ratio=ratio, value=value),
                transforms.ToPILImage()
            ])
random_erasing = transform(img)
random_erasing.save('random_erasing.jpg')

