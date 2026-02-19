# Darby Jones - 20260219

# script takes in nd2 files and converts them to tiff, does shape descriptor analysis
# and outputs csv with data and more tiff files with adjusted images, outlines, masks, and overlays
# to assist with analysis


import pandas as pd
import numpy as np
import os
import cv2
from nd2 import nd2_to_tiff
import tifffile as tiff
import re


#!!!!CHANGE THESE FOUR!!!!
rawFolder = '' #orignial nd2 files you wish to analyze
outputFolder = '' #ouput folder must already exist, all other folders will be made for you
microPerPixel = 2.64 # This will change based on objective (2 dimensional microns per pixel)
oneDMicronPerPixel = 1.62481 # This will change based on objective (1 dimensional microns per pixel)



# Constants for min and max vals based on 16-bit image and desired adjusted values (used during high contrast adjustment)
minVal = 0
maxVal = 65535
adjMax = 60000




# custom CLAHE func to maintain 16-bit data using 8-bit cv function (used for image adjustment)
# clip_limit is threshold for contrast limiting, grid size is size of grid for histogram (standard 8, 8)
def clahe16(image, clipLimitVar=4.0, gridSizeVar=(8, 8)):
    # normalize image to 16-bit range [0, 65535] (just in case)
    image = np.clip(image, 0, 65535).astype(np.uint16)

    # convert to float32 necessary for CLAHE processing (previously likley an int)
    imFloat = image.astype(np.float32)

    # normalize to 0-1 range to maintain data during change to 8-bit
    imNorm = imFloat / 65535.0

    # convert to 8-bit [0, 255] so can use cv CLAHE function
    im8bit = (imNorm * 255).astype(np.uint8)

    # create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clipLimitVar, tileGridSize=gridSizeVar)

    # apply CLAHE to 8 bit version of image
    imClahe8bit = clahe.apply(im8bit)

    # convert back to 16-bit [0, 65535] rounding to maintain precision
    imClahe16bit = np.round((imClahe8bit.astype(np.float32) / 255.0) * 65535).astype(np.uint16)

    return imClahe16bit


# function to convert nd2 to tif
def nd2ToTifConverstion(inputND2Folder, outputTifFolder):
    # Ensure output folder exists
    os.makedirs(outputTifFolder, exist_ok=True)

    # loop through nd2 files in selected folder
    for fileName in os.listdir(inputND2Folder):
        if fileName.endswith('.nd2'):
            inputPath = os.path.join(inputND2Folder, fileName)

            baseName, ext = os.path.splitext(fileName)
            newFileName = f'{baseName}_converted.tif'
            convertedOutputPath = os.path.join(outputTifFolder, newFileName)

            # convert ND2 to TIF
            try:
                nd2_to_tiff(inputPath, convertedOutputPath)
                print(f'Converted -- TIF saved as {newFileName}')
            except Exception as e:
                print(f'Error during conversion: {e}')



# function to adjust image color
def adjustImages(inputTifFolder, outputAdjustFolder, outputChannelFolder, brightenFactor=10):
    # Ensure output folder exists
    os.makedirs(outputAdjustFolder, exist_ok=True)

    # loop through tif files in selected folder
    for fileName in os.listdir(inputTifFolder):
        if fileName.endswith('tif'):
            inputPath = os.path.join(inputTifFolder, fileName)

            baseName, ext = os.path.splitext(fileName)
            newFileName = f'{baseName}_adjusted{ext}'

            adjustedOutputPath = os.path.join(outputAdjustFolder, newFileName)

            stack = tiff.imread(inputPath)
            print(stack.shape)
            # Check number of channels
            if stack.ndim == 3:
                numChannels = stack.shape[0]
                if numChannels >= 2:
                    # Picks 3rd channel for reading (generally DAPI)
                    selectedChannel = stack[1]  # third channel
                    print("Using channel 2")
                    channelFileName = f'{baseName}_channel2{ext}'
                else:
                    selectedChannel = stack[0]  # first channel
                    print("Using channel 1 (fallback)")
                    channelFileName = f'{baseName}_channel1{ext}'
            elif stack.ndim == 2:
                selectedChannel = stack  # single-channel image
                print("Single-channel image detected")
                channelFileName = f'{baseName}_channelOnly{ext}'
            else:
                raise ValueError(f"Unexpected image shape: {stack.shape}")

            channelOutputPath = os.path.join(outputChannelFolder, channelFileName)
            tiff.imwrite(channelOutputPath, selectedChannel)

            # read image (into array maintining 16 bit)
            im = cv2.imread(channelOutputPath, cv2.IMREAD_UNCHANGED)

            # check if converted to array
            if im is None:
                print(f'Error: Could not read the to adjust image {fileName}. Skipping...')
            else:
                print(f'Image for adjustment loaded with shape {im.shape} and dtype {im.dtype}.')

            # lighten whole image
            #im *= brightenFactor
            im = (im.astype(np.float32) * brightenFactor).clip(0, 65535).astype(np.uint16)

            # apply clahe
            adjustedIm = np.clip(clahe16(im), minVal, maxVal)

            # make sure all pixels in bounds
            im = np.clip(adjustedIm, minVal, maxVal)

            # save image
            cv2.imwrite(adjustedOutputPath, im)


# function to adjust image to high contrast, maybe useless
def highContrastAdjustImages(inputTifFolder, outputHcFolder, brightenFactor=10):
    # Ensure output folder exists
    os.makedirs(outputHcFolder, exist_ok=True)

    # loop through tif in selected folder
    for fileName in os.listdir(inputTifFolder):
        if fileName.endswith('tif'):
            inputPath = os.path.join(inputTifFolder, fileName)

            baseName, ext = os.path.splitext(fileName)
            newFileName = f'{baseName}_highConstrast{ext}'
            hcOutputPath = os.path.join(outputHcFolder, newFileName)

            # read image (into array)
            im = cv2.imread(inputPath, cv2.IMREAD_UNCHANGED)

            # check if converted to array
            if im is None:
                print(f'Error: Could not read the to high contrast image {fileName}. Skipping...')
            else:
                print(f'Image for high contrast loaded with shape {im.shape} and dtype {im.dtype}.')

            # lighten whole image
            im *= brightenFactor

            # lighten lightest pixels
            im[im > adjMax] -= 10000

            # make sure all pixels in bounds
            im = np.clip(im, minVal, maxVal)

            # save hc image
            cv2.imwrite(hcOutputPath, im)


def adaptiveThresholdSeg(inputTifFolder, outputMaskFolder,
                         regionSize=1001, cValue=-1, minArea=4000,
                         min_solidity=0.2, max_aspect=12):
    """
    Adaptive threshold with area and shape filtering.

    Parameters:
    -----------
    regionSize : int
        Size of neighborhood for adaptive threshold (must be odd)
    cValue : int
        Constant subtracted from mean
    minArea : int
        Minimum area to keep (pixels)
    min_solidity : float
        Minimum solidity (area/convex_hull_area) to filter debris
    max_aspect : float
        Maximum aspect ratio to filter stringy debris
    """
    os.makedirs(outputMaskFolder, exist_ok=True)

    for fileName in os.listdir(inputTifFolder):
        if fileName.endswith('tif'):
            inputPath = os.path.join(inputTifFolder, fileName)

            baseName, ext = os.path.splitext(fileName)
            newFileName = f'{baseName}_mask{ext}'
            adjustedOutputPath = os.path.join(outputMaskFolder, newFileName)

            # read image (into array)
            image = cv2.imread(inputPath, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f'Error: Could not read the to adaptive threshold image {fileName}. Skipping...')
                continue

            # slight blur to reduce noise
            blurredImage = cv2.GaussianBlur(image, (5, 5), 0)

            # normalize image to 16-bit range [0, 65535] (just in case)
            image_clip = np.clip(blurredImage, 0, 65535).astype(np.uint16)
            # convert to float32 necessary for CLAHE processing (previously likely an int)
            imFloat = image_clip.astype(np.float32)
            # normalize to 0-1 range to maintain data during change to 8-bit
            imNorm = imFloat / 65535.0
            # convert to 8-bit [0, 255] so can use cv CLAHE function
            im8bit = (imNorm * 255).astype(np.uint8)

            # apply threshold - ARITHMETIC VERSION
            maxVal = 255
            mask = cv2.adaptiveThreshold(im8bit, maxVal, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, regionSize, cValue)

            # ===== AREA AND SHAPE FILTERING =====
            # Find connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

            # Create clean mask
            clean_mask = np.zeros_like(mask)
            kept = 0

            for i in range(1, num_labels):  # Skip background (label 0)
                area = stats[i, cv2.CC_STAT_AREA]

                # Filter by minimum area
                if area < minArea:
                    continue

                # Get object mask
                obj_mask = (labels == i).astype(np.uint8) * 255

                # Find contour
                contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

                if not contours:
                    continue

                contour = contours[0]

                # Calculate solidity (compactness)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0

                # Calculate aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 0

                # Keep if passes filters
                if solidity >= min_solidity and aspect <= max_aspect:
                    clean_mask[labels == i] = 255
                    kept += 1

            print(f'{fileName}: Kept {kept} / {num_labels - 1} objects')

            # save image
            cv2.imwrite(adjustedOutputPath, clean_mask)

            print('mask complete')


def outlineObj(inputMaskFolder, adjustedFolder, outputNumberFolder, outputCSV, minArea=400):
    # naming and reading batch
    os.makedirs(outputNumberFolder, exist_ok=True)

    # Create df to catch all image data in the folder
    folderData = []

    for fileName in os.listdir(inputMaskFolder):
        if fileName.endswith('tif'):
            inputPath = os.path.join(inputMaskFolder, fileName)
            baseName, ext = os.path.splitext(fileName)
            newFileName = f'{baseName}_numbered{ext}'
            adjustedOutputPath = os.path.join(outputNumberFolder, newFileName)

            im = cv2.imread(inputPath, cv2.IMREAD_UNCHANGED)
            match = re.match(r'(.+)_channel\d+_mask\.tif$', fileName)

            if not match:
                raise ValueError(f"Unexpected mask filename format: {fileName}")

            baseStem = match.group(1)

            # rebuild adjusted filename
            adjustedName = f'{baseStem}_adjusted.tif'
            adjustedPath = os.path.join(adjustedFolder, adjustedName)

            adjustedImage = cv2.imread(adjustedPath, cv2.IMREAD_UNCHANGED)

            if len(adjustedImage.shape) == 2:
                overlayImage = cv2.cvtColor(adjustedImage, cv2.COLOR_GRAY2BGR)
            else:
                overlayImage = adjustedImage.copy()

            # Read mask as-is (should already be uint8 0/255)
            mask8bit = im.copy()

            # Ensure correct dtype
            if mask8bit.dtype != np.uint8:
                mask8bit = mask8bit.astype(np.uint8)

            # Create a blank canvas for drawing outlines
            outlinedImage = np.zeros((*mask8bit.shape, 3), dtype=np.uint8)

            print(
                "mask dtype:", mask8bit.dtype,
                "unique values:", np.unique(mask8bit)
            )

            # Find contours; cv2.RETR_EXTERNAL means that it ignores internal holes and things; cv2.CHAIN_APPROX_SIMPLE saves mem
            contours, _ = cv2.findContours(mask8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            objectIndex = 1  # Start numbering from 1

            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= minArea:  # Ignore small objects
                    # Draw contours for outline and overlay
                    cv2.drawContours(outlinedImage, [contour], -1, (255,255,255), thickness=2)
                    cv2.drawContours(overlayImage, [contour], -1, (0, 0, 255), thickness=2)

                    # Compute centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:  # Avoid division by zero
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

                        # Draw number (ensuring it's inside the object)
                        cv2.putText(outlinedImage, str(objectIndex), (cX, cY),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,255,0), 1, cv2.LINE_AA)
                        # number on overlay (red)
                        cv2.putText(overlayImage, str(objectIndex), (cX, cY),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    # COLLECTING DATA
                    perimeter = cv2.arcLength(contour, closed=True)
                    inverseCircularity = (perimeter ** 2) / (4 * np.pi * area)
                    circularity = (4 * np.pi * area) / (perimeter ** 2)
                    roundness = None
                    aspectRatio = None

                    # Calculations that need fitted shape (fit oval, not rectangle)
                    if len(contour) >= 5:  # fitEllipse needs at least 5 points
                        ellipse = cv2.fitEllipse(contour)
                        (_, axes, _) = ellipse
                        majorAxis = max(axes)
                        minorAxis = min(axes)
                        aspectRatio = majorAxis / minorAxis
                        roundness = 4 * area / (np.pi * majorAxis ** 2)

                    # Solidity (Convex Hull)
                    hull = cv2.convexHull(contour)
                    hullArea = cv2.contourArea(hull)
                    solidity = area / hullArea if hullArea > 0 else None

                    # pixel to micron unit conversion
                    micronArea = area * (microPerPixel ** 2)
                    micronPerimeter = perimeter * microPerPixel


                    # Store data
                    folderData.append(
                        [objectIndex, fileName, micronArea, micronPerimeter, circularity, inverseCircularity,
                         aspectRatio, roundness, solidity, cX*oneDMicronPerPixel, cY*oneDMicronPerPixel, majorAxis])

                    objectIndex += 1

            outline16bit = np.round((outlinedImage.astype(np.float32) / 255.0) * 65535).astype(np.uint16)
            invertedImage = cv2.bitwise_not(outline16bit)

            cv2.imwrite(adjustedOutputPath, invertedImage)
            # Convert overlay to 8-bit if needed
            overlay8 = overlayImage
            if overlay8.dtype != np.uint8:
                overlay8 = cv2.normalize(
                    overlay8, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)

            overlayPath = os.path.join(outputNumberFolder, f'{baseName}_overlay.png')
            cv2.imwrite(overlayPath, overlay8)
            print(f'Saved outline image for {fileName}')


    # Create DataFrame
    df = pd.DataFrame(folderData,
                      columns=['Organoid Index', 'Label', 'Area (microns)', 'Perimeter (microns)', 'Circularity',
                               'Inverse Circularity', 'AR', 'Roundness', 'Solidity', 'cX', 'cY', "MAJOR AXIS SIR!!!"])

    # round to 3 dec places
    df = df.round(3)

    # Save to CSV
    df.to_csv(outputCSV, index=False)

    print(f'All object measurements saved to {outputCSV}')



# assigns conditions by prompting user once per file
def assignConditions(maskFolder, csvPath, predefMapping=None):
    tifFiles = [f for f in os.listdir(maskFolder) if f.endswith('.tif')]
    if predefMapping is None:
        print("\nDetected the following TIF files:")
        for f in tifFiles:
            print(f" - {f}")

        print("\nNow enter a condition label for each file individually.")

        conditions = {}
        for file in tifFiles:
            conditionLabel = input(f"What condition should be assigned to:\n  {file}\n> ").strip()
            conditions[file] = conditionLabel
    else:
         conditions = predefMapping
         print(f"\nUsing predefined mapping:\n{conditions}")

    # Load the existing CSV
    df = pd.read_csv(csvPath)

    # Assign condition based on substrings in filename
    def assignCon(conditionLabel):
        return conditions.get(conditionLabel, "unknown")

    df["Condition"] = df["Label"].apply(assignCon)

    # Desired column order: Organoid Index, Label, Condition, ...
    desiredCols = ['Organoid Index', 'Label', 'Condition']

    # Add remaining columns in current order
    remainingCols = [col for col in df.columns if col not in desiredCols]
    df = df[desiredCols + remainingCols]

    # Save updated DataFrame
    df.to_csv(csvPath, index=False)
    print(f"\nCondition labels added and saved to: {csvPath}")


def processImages(inputND2Folder, outputTifFolder, outputAdjFolder, outputChannelFolder, outputHcFolder,
                  outputMaskFolder, outputNumberFolder, outputCSV):
    nd2ToTifConverstion(inputND2Folder, outputTifFolder)
    adjustImages(outputTifFolder, outputAdjFolder, outputChannelFolder)
    highContrastAdjustImages(outputChannelFolder, outputHcFolder)
    adaptiveThresholdSeg(outputChannelFolder, outputMaskFolder,
                         regionSize=1001,
                         cValue=-1,
                         minArea=4000,
                         min_solidity=0.2,
                         max_aspect=12)
    outlineObj(outputMaskFolder, outputAdjFolder, outputNumberFolder, outputCSV)


def makeFolders(motherFolder, rawImages):
    tiff = os.path.join(motherFolder, "tiff")
    adjusted = os.path.join(motherFolder, "adjusted")
    highContrast = os.path.join(motherFolder, "highContrast")
    masks = os.path.join(motherFolder, "mask")
    numberedOutlines = os.path.join(motherFolder, "outlines")
    analysis = os.path.join(motherFolder, "analysis")
    data = os.path.join(motherFolder, "data")

    initialData = os.path.join(data, "initial.csv")

    os.makedirs(tiff, exist_ok=True)
    os.makedirs(adjusted, exist_ok=True)
    os.makedirs(highContrast, exist_ok=True)
    os.makedirs(masks, exist_ok=True)
    os.makedirs(numberedOutlines, exist_ok=True)
    os.makedirs(analysis, exist_ok=True)
    os.makedirs(data, exist_ok=True)

    processImages(rawImages, tiff, adjusted, analysis, highContrast,
                  masks, numberedOutlines, initialData)

    assignConditions(masks, initialData)

# run prossesing using user inputed folder locations
makeFolders(outputFolder, rawFolder)
