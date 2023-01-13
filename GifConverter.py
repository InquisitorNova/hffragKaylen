#Import Relevant modules
from PIL import Image
import os, sys
import glob 

#Moves directory to the file containing the predictions per epoch
Directory = '/home/physics/phujdj/DeepLearningParticlePhysics/EpochPlots'
os.chdir(Directory)

#Creates a txt file for gif command
frames = []
fileList = glob.glob("*.png")
for image in fileList:
    new_frame = Image.open(image)
    frames.append(new_frame)

#Save into a gif file that loops forever
frames[0].save('/home/physics/phujdj/DeepLearningParticlePhysics/EpochPlots.gif', format = "GIF",
                append_images = frames[1:],
                save_all = True,
                duration = 300, loop = 1)
