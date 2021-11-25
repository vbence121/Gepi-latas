#from os import terminal_size
#from numpy.core.fromnumeric import size
from types import NoneType
import cv2 as cv
import imutils
import numpy as np
import pytesseract
import getopt
import sys
import os
from pathlib import Path
import re
from pytesseract.pytesseract import main

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\TIBDBQN\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
verbose = False     # if enabled images are displayed while detection is in progress. 
verboseF = False    # if enabled images are displayed while detection is in progress (they are all displayed at the same time). 
helpMSG = "detect.py -i <inputfile> [-v/-V] [-b] [-S]"
batchM = False      # enable batch mode
silentM = False     # enable silent mode
patternM = False    # enable pattern based otr 

inputfile = r"" # file to be read
Ffilter = "-" # filter used on result when comparing to expected output
Gsize = (600,400)

def patternSearch(inp:str)->str:
    if(inp == NoneType):
        return ""
    if re.search(r"[A-Z][A-Z][A-Z][0-9][0-9][0-9]", str.upper(inp)):
        matches = re.search(r"[A-Z][A-Z][A-Z][0-9][0-9][0-9]", str.upper(inp))
        return matches.group()

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

    print("exp:"+str.upper(expct).strip()+" result:"+str.upper(res).strip())
        
    if(res != "" or type(res) != NoneType) and (str.upper(expct).strip() == str.upper(res).strip()):
        return True
    else:   # if failed try different pre-process
        if((len(res) > 6 ) and (str.upper(expct).strip() == str.upper(patternSearch(res)))):
            return True
        res = detect(file, eBlur=False, eTresshold=True, pBlur=True, cMethod=cv.CHAIN_APPROX_TC89_KCOS)
        for s in Ffilter:
            res = res.replace(s,"")
        if(res != "" or res != NoneType) and (str.upper(expct).strip() == str.upper(res).strip()):
            return True
        else:
            if((patternSearch(res) != "" and type(patternSearch(res)) != NoneType) and ((len(res) > 6 ) and (str.upper(expct).strip() == str.upper(patternSearch(res))))):
                return True
            if(patternM):
                res = detect(file, ePattern=True)
                for s in Ffilter:
                    res = res.replace(s,"")
                if re.search(r"[A-Z][A-Z][A-Z][0-9][0-9][0-9]", str.upper(res)):
                    matches = re.search(r"[A-Z][A-Z][A-Z][0-9][0-9][0-9]", str.upper(res))
                    print("Pattern based search result: "+matches.group())
                    if(res != "" or res != NoneType) and (str.upper(expct).strip() == str.upper(matches.group()).strip()):
                        return True
                    return False
            else:
                return False


def detect(file:str, eTresshold:bool=False, eBlur:bool=True, pBlur:bool=False, cMethod=cv.CHAIN_APPROX_SIMPLE, fSigma:int=15, ePattern:bool= False) -> str:
    if(not(Path(file).exists()) or file == ""):
        print("File at given path does not exist.")
        sys.exit()

    img = cv.imread(file)
    img = cv.resize(img, (600,400))

    if(verbose or verboseF):
        cv.imshow("Color", img)
    if(not (verboseF)):
        if(cv.waitKey(0) == ord('a')):
            sys.exit(1)
        cv.destroyAllWindows()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    gray = cv.bilateralFilter(gray, 13, fSigma, fSigma)

    if(verbose or verboseF):
        cv.imshow("Gray", gray)
    if(not (verboseF)):
        if(cv.waitKey(0) == ord('a')):
            sys.exit(1)
        cv.destroyAllWindows()

    if(pBlur):
        gray = cv.medianBlur(gray,3)

    if(ePattern):
        text = pytesseract.image_to_string(gray, config='--psm 11')
        text = re.sub(r"[^A-Z0-9]", "", text)
        if(text != ""):
            print("Detected license plate number is (Pattern based):",text)
        else:
            print("License plate number could not be detected (Pattern based)!")
        if(verbose or verboseF  ):
            if(cv.waitKey(0) == ord('a')):
                sys.exit(1)
            cv.destroyAllWindows()
        return text

    edged = cv.Canny(gray, 75, 250) 
    contours = cv.findContours(edged.copy(), cv.RETR_TREE, cMethod)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv.contourArea, reverse = True)
    screenCnt = None

    if(verbose or verboseF):
        cv.imshow("Contour", edged)
    #if(not (verboseF)):
        if(cv.waitKey(0) == ord('a')):
            sys.exit(1)
        cv.destroyAllWindows()

    maxArea = cv.contourArea(contours[0])
    for c in contours:
        
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.020 * peri, True)
        if len(approx) == 4:
            if(maxArea*0.9 > cv.contourArea(c)):
                break
            if(maxArea < cv.contourArea(c)):
                maxArea = cv.contourArea(c)
            screenCnt = approx
            break
    if screenCnt is None:
        print ("No contour detected")
        return ""
    
    cv.drawContours(img, [screenCnt], -1, (0, 0, 255), 3) # draw contour around detected licence plate
    
    mask = np.zeros(gray.shape,np.uint8)
    mask = cv.drawContours(mask,[screenCnt],0,255,-1) # draw mask
    
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]  # cropp out masked area

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
    if(text != ""):
        print("Detected license plate number is:",text)
    else:
        print("License plate number could not be detected!")
    if(not(silentM)):
        img = cv.resize(img,(500,300))
        Cropped = cv.resize(Cropped,(400,200))
        cv.imshow('Original',img)
        cv.imshow('Cropped',Cropped)
        if(cv.waitKey(0) == 27):
            sys.exit(1)
        cv.destroyAllWindows()
    return text


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:s:VvbBSsMm")
    except getopt.getopt.GetoptError:
        print(helpMSG)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(helpMSG)
            sys.exit()
        elif opt in ("-i"):
            global inputfile 
            inputfile = arg.strip()
            if(not(Path(arg).exists()) and not(Path(inputfile).is_dir())):
                print("File at given path does not exist.")
                sys.exit(-1)
        elif opt == "-v":
            global verbose 
            verbose = True
        elif opt == "-V":
            global verboseF 
            verboseF = True
        elif (opt == "-b" or opt == "-B"):
            global batchM 
            batchM = True
        elif (opt == "-S" or opt in "-s"):
            global silentM 
            silentM = True
        elif (opt == "-M" or opt == "-m"):
            global patternM
            patternM = True

    if(batchM and Path(inputfile).is_dir()):
        correct = 0
        all = len([name for name in os.listdir(inputfile) if os.path.isfile(inputfile+name)])
        for file in os.listdir(inputfile):
            if(not(Path(inputfile+file).is_file())):
                continue
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