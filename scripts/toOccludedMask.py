import cv2
import os



fullfiles = (os.listdir("MASK DIRECTORY"))
visibfiles = (os.listdir("MASK DIRECTORY"))
outPath = "OUTPATH"
for i in range(len(fullfiles)):
    full1 = fullfiles  + fullfiles[i]
    visib1 = visibfiles + fullfiles[i]

    full = cv2.imread(full1)
    visib = cv2.imread(visib1)


    subtracted = cv2.subtract(full, visib)
    subtracted = cv2.cvtColor(subtracted, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("OUTPATH"+fullfiles[i],subtracted)
    # cv2.imshow('image', subtracted)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

