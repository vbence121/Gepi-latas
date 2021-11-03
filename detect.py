#from os import terminal_size
#from numpy.core.fromnumeric import size
import cv2 as cv
import imutils
import numpy as np
import pytesseract
import getopt
import sys
from pathlib import Path
import re
from pytesseract.pytesseract import main

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\TIBDBQN\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
verbose = False     # if enabled images are displayed while detection is in progress. 
verboseF = False    # if enabled images are displayed while detection is in progress (they are all displayed at the same time). 
helpMSG = "detect.py -i <inputfile> [-v/-V] [-b] [-S]"
batchM = False      # enable batch mode
silentM = False     # enable silent mode

Ffilter = "-" # filter used on result when comparing to expected output
Gsize = (600,400)

def match(file:str) -> bool:
    """
    Try to read image file and check result against expected license plate.
    """
    if(file == ""):
        print(helpMSG)
    if(not(Path(file).exists())):
        print("File at given path does not exist.")
        sys.exit(-1)
    res = detect(file)

    expct = Path(file).stem
    for s in Ffilter:
        res = res.replace(s,"")

    if(batchM):
        print("exp:"+str.upper(expct).strip()+" result:"+str.upper(res).strip(),end="") # cleanup print based on mode
    else:
        print("exp:"+str.upper(expct).strip()+" result:"+str.upper(res).strip())
        
    if(str.upper(expct).strip() == str.upper(res).strip()):
        return True
    else:   # if failed try different pre-process
        res = detect(file, eBlur=False, eTresshold=True, pBlur=True, cMethod=cv.CHAIN_APPROX_TC89_KCOS)
        for s in Ffilter:
            res = res.replace(s,"")
            if(str.upper(expct).strip() == str.upper(res).strip()):
                return True
            else:
                return False


def detect(file:str, eTresshold:bool=False, eBlur:bool=True, pBlur:bool=False, cMethod=cv.CHAIN_APPROX_SIMPLE, fSigma:int=15) -> str:
    if(not(Path(file).exists()) or file == ""):
        print("File at given path does not exist.")
    img = cv.imread(file)
    img = cv.resize(img, (600,400))
    if(verbose or verboseF):
        cv.imshow("Color", img)
    if(not (verboseF)):
        cv.waitKey(0)
        cv.destroyAllWindows()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    gray = cv.bilateralFilter(gray, 13, fSigma, fSigma)
    if(verbose or verboseF):
        cv.imshow("Gray", gray)
    if(not (verboseF)):
        cv.waitKey(0)
        cv.destroyAllWindows()
    if(pBlur):
        gray = cv.medianBlur(gray,3)

    edged = cv.Canny(gray, 75, 250) 
    contours = cv.findContours(edged.copy(), cv.RETR_TREE, cMethod)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv.contourArea, reverse = True)
    screenCnt = None

    if(verbose or verboseF):
        cv.imshow("Contour", edged)
    #if(not (verboseF)):
        cv.waitKey(0)
        cv.destroyAllWindows()

    for c in contours:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.020 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt is None:
        detected = 0
        print ("No contour detected")
        return ""
    
    cv.drawContours(img, [screenCnt], -1, (0, 0, 255), 3) # draw contour around detected licence plate
    
    mask = np.zeros(gray.shape,np.uint8)
    mask = cv.drawContours(mask,[screenCnt],0,255,-1) # draw mask
    #mask = cv.bitwise_and(img,img,mask=mask)
    
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]

    if(eBlur):
        Cropped = cv.medianBlur(Cropped,3)
    #Cropped = cv.bilateralFilter(Cropped,9,75,75)
    
    if(eTresshold):
        kernel = np.ones((3,3),np.uint8)
        Cropped = cv.morphologyEx(Cropped, cv.MORPH_OPEN, kernel)
        #Cropped = cv.threshold(Cropped, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
        Cropped = cv.adaptiveThreshold(Cropped,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,5,2)
        #Cropped = cv.dilate(Cropped, kernel, iterations = 1)

    text = pytesseract.image_to_string(Cropped, config='--psm 11')
    text = re.sub(r"[^A-Z0-9]", "", text)
    print("Detected license plate Number is:",text)
    img = cv.resize(img,(500,300))
    Cropped = cv.resize(Cropped,(400,200))
    if(not(silentM)):
        cv.imshow('Original',img)
        cv.imshow('Cropped',Cropped)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return text


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:s:VvbS")
    except getopt.getopt.GetoptError:
        print(helpMSG)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(helpMSG)
            sys.exit()
        elif opt in ("-i"):
            global inputfile 
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
        elif opt == "-b":
            global batchM 
            batchM = True
        elif opt == "-S":
            global silentM 
            silentM = True

    if(batchM and Path(inputfile).is_dir()):
        correct = 0
            if(match(inputfile+file)):
                correct = correct+1
                print(" Matching!")
            else:
                print(" Result does not match or image detection failed.")
        print("\nMatch ratio: "+str(correct)+"\\"+str(all))
    else:
        if(match(inputfile)):
            print("Matching!")
        else:
            print("Result does not match or image detection failed.")


if __name__ == "__main__":
   main()