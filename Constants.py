import os
import pygame

modelname = "racer_9state_3" + ".zip"

# Window CONFIGURE
WIDTH, HEIGHT = 550, 550
# TIME_LIMIT    = 60  #How many seconds will it take for one episode?

# Model.learn - Hyperparameter Configure
total_timesteps = 150000 #300k
learning_rate  = 0.0005 #0.004 (4*10^-3) recommended
ent_coef       = 0.01
gamma          = 0.99 
gae_lambda     = 0.95
max_grad_norm  = 0.5

# Physical CONSTANTS
FPS         = 20

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
INDIGO = (75, 0, 130)
VIOLET = (238, 130, 238)

# Model Configure
Orig_Model_Save_Path = "./models/" + modelname
# print("Your model will be saved as: " + Model_Save_Path)


def get_updated_Model_Save_Path(modelname):
    import re
    while os.path.exists("./models/" + modelname):
        try:
            start_idx = (modelname.find(re.search(r'\d+', modelname[::-1]).group()[::-1]))
            num = int(re.search(r'\d+', modelname[::-1]).group()[::-1])
            num_len = (len(re.search(r'\d+', modelname[::-1]).group()[::-1]))
            modelname = modelname[:start_idx] + str(num+1) + modelname[start_idx+num_len:]
        except:
            i = modelname.find(".")
            modelname = modelname[:i] + str(1) + modelname[i:]
        
    Model_Save_Path = "./models/" + modelname
    print("Your model will be saved as: " + Model_Save_Path)
    return Model_Save_Path



tensorboard_log = None      # "./DroneLog/"
tensorboard_sub_folder = None       # 'new_training' + str(total_timesteps/1000) + "k"

# Display and asset Settings & Function
# BACKGROUND = "assets/sky.png"

# Takes multiple image and provide animation in the game
CAR_WIDTH = 80/10
CAR_HEIGHT = (8/30) * 621 / 10
def spriter(Type):
    if Type == "Car":
        image_width = CAR_WIDTH
        image_height = CAR_HEIGHT
        image_path = "./assets/"

    player = []

    for image_file in os.listdir(image_path):
        file_path = os.path.join(image_path, image_file)  # Full path to the file
        if os.path.isfile(file_path) and image_file.lower().endswith('.png'):
            image = pygame.image.load(file_path)
            image.convert()
            player.append(pygame.transform.scale(image, (image_width, image_height)))

    return player

def report(environment):
    print("Environment loading..\n")
    print("Observation space:")
    print(environment.observation_space)
    print("")
    print("Action space:")
    print(environment.action_space)
    print("")
    print("Action space sample:")
    print(environment.action_space.sample())
