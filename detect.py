#from os import terminal_size
#from numpy.core.fromnumeric import size
import cv2 as cv
import imutils
import numpy as np
import pytesseract
import getopt
import sys
from pathlib import Path

from pytesseract.pytesseract import main

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
verbose = False     # if enabled images are displayed while detection is in progress. 
verboseF = False    # if enabled images are displayed while detection is in progress (they are all displayed at the same time). 
helpMSG = "detect.py -i <inputfile> [-v/-V]"

inputfile = "" # file to be read
Ffilter = "-" # filter used on result when comparing to expected output
Gsize = (600,400)

def match() -> bool:
    """
    Try to read image file and check result against expected license plate.
    """
    if(inputfile == ""):
        print(helpMSG)
    if(not(Path(inputfile).exists())):
        print("File at given path does not exist.")
        sys.exit(-1)
    res = detect()

    expct = Path(inputfile).stem
    for s in Ffilter:
        res = res.replace(s,"")
    print("exp:"+str.upper(expct).strip()+" result:"+str.upper(res).strip())
    if(str.upper(expct).strip() == str.upper(res).strip()):
        return True
    else:
        return False


def detect() -> str:
    if(not(Path(inputfile).exists()) or inputfile == ""):
        print("File at given path does not exist.")
        sys.exit(-1)
    img = cv.imread(inputfile)
    img = cv.resize(img, (600,400))
    if(verbose or verboseF):
        cv.imshow("Color", img)
    if(not (verboseF)):
        cv.waitKey(0)
        cv.destroyAllWindows()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    gray = cv.bilateralFilter(gray, 13, 15, 15)
    if(verbose or verboseF):
        cv.imshow("Gray", gray)
    if(not (verboseF)):
        cv.waitKey(0)
        cv.destroyAllWindows()

    edged = cv.Canny(gray, 30, 200) 
    contours = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv.contourArea, reverse = True)[:10]
    screenCnt = None
    if(verbose or verboseF):
        cv.imshow("Contour", edged)
    #if(not (verboseF)):
        cv.waitKey(0)
        cv.destroyAllWindows()

    for c in contours:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.018 * peri, True)
    
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        detected = 0
        print ("No contour detected")
        return ""
    
    cv.drawContours(img, [screenCnt], -1, (0, 0, 255), 3) # draw contour around detected licence plate
    
    mask = np.zeros(gray.shape,np.uint8)
    mask = cv.drawContours(mask,[screenCnt],0,255,-1,) # draw mask
    #mask = cv.bitwise_and(img,img,mask=mask)
    
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]

    text = pytesseract.image_to_string(Cropped, config='--psm 11')
    print("Detected license plate Number is:",text[:-1].strip())
    img = cv.resize(img,(500,300))
    Cropped = cv.resize(Cropped,(400,200))
    cv.imshow('Original',img)
    cv.imshow('Cropped',Cropped)

    cv.waitKey(0)
    cv.destroyAllWindows()
    return text[:-1]


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:s:Vv")
    except getopt.getopt.GetoptError:
        print(helpMSG)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(helpMSG)
            sys.exit()
        elif opt in ("-i"):
            global inputfile 
            inputfile = arg
            if(not(Path(arg).exists())):
                print("File at given path does not exist.")
                sys.exit(-1)
        elif opt in ("-s"):
            global Gsize
            Gsize = arg
        elif opt == "-v":
            global verbose 
            verbose = True
        elif opt == "-V":
            global verboseF 
            verboseF = True

    if(match()):
        print("Matching!")
    else:
        print("Result does not match or image detection failed.")


if __name__ == "__main__":
   main()