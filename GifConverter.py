"""
*Filename: GifConverter
*Description: This python files converts the images of the 
*true vs predicted scatterplot per epoch and converts them into a gif that 
*shows how the model predictions improve with time.
Date: 16/02/2023
Author: Kaylen Smith Darnbrook
"""

#Import Relevant modules
from PIL import Image
import os, sys
import glob 

Epoch_Number = 169
#Moves directory to the file containing the predictions per epoch
Directory = '/home/physics/phujdj/DeepLearningParticlePhysics/EpochPlotsCorrected'
os.chdir(Directory)

#Creates a txt file for gif command
frames = []
fileList  = []
#fileList = glob.glob("*.png")
for file_number in range(Epoch_Number):
    fileList.append("PxPredictionOnEpoch-{0}.png".format(file_number))

print(fileList)
for image in fileList:
    new_frame = Image.open(image)
    frames.append(new_frame)
 
#Save into a gif file that loops forever
frames[0].save('/home/physics/phujdj/DeepLearningParticlePhysics/EpochPlots.gif', format = "GIF",
                append_images = frames[1:],
                save_all = True,
                duration = 300, loop = 1)
