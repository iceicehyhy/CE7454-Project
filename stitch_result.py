
import PIL.Image as Image
import os

IMAGES_PATH_original = './data/ori_test/'
IMAGES_PATH_masked_ori = './data/masked_pic_test/'
IMAGES_PATH_masked_ce = './results/context_encoder_input/'
IMAGES_PATH_bak = './results/background/'
IMAGES_PATH_context_enc = './results/context_encoder/'
IMAGES_PATH_GLCIC = './results/GLCIC/'
IMAGES_PATH_attention = './results/contextual_attention/'
IMAGES_PATH_ours = './results/proposed/'

IMAGES_FORMAT = ['.jpg', '.JPG', '.png', '.PNG']
IMAGE_SIZE = 128
IMAGE_ROW = 6
IMAGE_COLUMN = 7
IMAGE_SAVE_PATH = './results/final.jpg'
 
# get all pics
image_names_ori = [name for name in os.listdir(IMAGES_PATH_original) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]

image_names_masked_ce = [name for name in os.listdir(IMAGES_PATH_masked_ori) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]
image_names_masked_ori = [name for name in os.listdir(IMAGES_PATH_masked_ori) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]

image_names_bak = [name for name in os.listdir(IMAGES_PATH_bak) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]

image_names_context_enc = [name for name in os.listdir(IMAGES_PATH_context_enc) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]
image_names_GLCIC = [name for name in os.listdir(IMAGES_PATH_GLCIC) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]
image_names_attention = [name for name in os.listdir(IMAGES_PATH_attention) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]
image_names_ours = [name for name in os.listdir(IMAGES_PATH_ours) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]          

 
# compose function
def image_compose():
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE)) # create a big picture
    # iterativelly search for pic to compose
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            if y == 1:
                from_image = Image.open(IMAGES_PATH_bak + str(x) + '.png').resize(
                    (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
                to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
            else:
                if x == 1:
                    from_image = Image.open(IMAGES_PATH_original + image_names_ori[y+4]).resize(
                        (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
                    to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
                elif x == 2:
                    from_image = Image.open(IMAGES_PATH_masked_ce + image_names_masked_ce[y+4]).resize(
                        (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
                    to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
                elif x == 3:
                    from_image = Image.open(IMAGES_PATH_context_enc + image_names_context_enc[y+4]).resize(
                        (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
                    to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
                elif x == 4:
                    from_image = Image.open(IMAGES_PATH_masked_ori + image_names_masked_ori[y+4]).resize(
                        (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
                    to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
                elif x == 5:
                    from_image = Image.open(IMAGES_PATH_GLCIC + image_names_GLCIC[y+4]).resize(
                        (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
                    to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
                elif x == 6:
                    from_image = Image.open(IMAGES_PATH_attention + image_names_attention[y+4]).resize(
                        (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
                    to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
                elif x == 7:
                    from_image = Image.open(IMAGES_PATH_ours + image_names_ours[y+4]).resize(
                        (IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
                    to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
    return to_image.save(IMAGE_SAVE_PATH)

image_compose()
