#!/usr/bin/python

############################################################
# ExtractLines.py
#
# This script reads in range data from a csv file, and
# implements a split-and-merge to extract meaningful lines
# in the environment.
############################################################

# Imports
import numpy as np
from PlotFunctions import *


############################################################
# functions
############################################################

def ExtractLines(RangeData, params):
    '''
    This function implements a split-and-merge line extraction algorithm

    INPUT: RangeData - (x_r, y_r, theta, rho)
                x_r - robot's x position (m)
                y_r - robot's y position (m)
              theta - (1D) np array of angle 'theta' from data (rads)
                rho - (1D) np array of distance 'rho' from data (m)
           params - dictionary of parameters for line extraction

    OUTPUT: (alpha, r, segend, pointIdx)
         alpha - (1D) np array of 'alpha' for each fitted line (rads)
             r - (1D) np array of 'r' for each fitted line (m)
        segend - np array (N_lines, 4) of line segment endpoints.
                 each row represents [x1, y1, x2, y2]
      pointIdx - (N_lines,2) segment's first and last point index
    '''

    # Extract useful variables from RangeData
    x_r = RangeData[0]
    y_r = RangeData[1]
    theta = RangeData[2]
    rho = RangeData[3]

    ### Split Lines ###
    N_pts = len(rho)
    r = np.zeros(0)
    alpha = np.zeros(0)
    pointIdx = np.zeros((0, 2), dtype=np.int)

    # This implementation pre-prepartitions the data according to the "MAX_P2P_DIST"
    # parameter. It forces line segmentation at sufficiently large range jumps.
    rho_diff = np.abs(rho[1:] - rho[:(len(rho)-1)])
    LineBreak = np.hstack((np.where(rho_diff > params['MAX_P2P_DIST'])[0]+1, N_pts))
    # LineBreak = np.arange(2,181,2)
    startIdx = 0
    for endIdx in LineBreak:
        alpha_seg, r_seg, pointIdx_seg = SplitLinesRecursive(theta, rho, startIdx, endIdx, params)
        N_lines = r_seg.size

        ### Merge Lines ###
        if (N_lines > 1):
             alpha_seg, r_seg, pointIdx_seg = MergeColinearNeigbors(theta, rho, alpha_seg, r_seg, pointIdx_seg, params)
        r = np.append(r, r_seg)
        alpha = np.append(alpha, alpha_seg)
        pointIdx = np.vstack((pointIdx, pointIdx_seg))
        startIdx = endIdx

    N_lines = alpha.size

    ### Compute endpoints/lengths of the segments ###
    segend = np.zeros((N_lines, 4))
    seglen = np.zeros(N_lines)
    for i in range(N_lines):
        rho1 = r[i]/np.cos(theta[pointIdx[i, 0]]-alpha[i])
        rho2 = r[i]/np.cos(theta[pointIdx[i, 1]-1]-alpha[i])
        x1 = rho1*np.cos(theta[pointIdx[i, 0]])
        y1 = rho1*np.sin(theta[pointIdx[i, 0]])
        x2 = rho2*np.cos(theta[pointIdx[i, 1]-1])
        y2 = rho2*np.sin(theta[pointIdx[i, 1]-1])
        segend[i, :] = np.hstack((x1, y1, x2, y2))
        seglen[i] = np.linalg.norm(segend[i, 0:2] - segend[i, 2:4])

    ### Filter Lines ###
    # Find and remove line segments that are too short
    goodSegIdx = np.where((seglen >= params['MIN_SEG_LENGTH']) &
                          (pointIdx[:, 1] - pointIdx[:, 0] >= params['MIN_POINTS_PER_SEGMENT']))[0]
    pointIdx = pointIdx[goodSegIdx, :]
    alpha = alpha[goodSegIdx]
    r = r[goodSegIdx]
    segend = segend[goodSegIdx, :]

    # change back to scene coordinates
    segend[:, (0, 2)] = segend[:, (0, 2)] + x_r
    segend[:, (1, 3)] = segend[:, (1, 3)] + y_r
    print(alpha)
    print(r)
    
    return alpha, r, segend, pointIdx


def SplitLinesRecursive(theta, rho, startIdx, endIdx, params):
    '''
    This function executes a recursive line-slitting algorithm, 
    which recursively sub-divides line segments until no further 
    splitting is required.

    INPUT:  theta - (1D) np array of angle 'theta' from data (rads)
           rho - (1D) np array of distance 'rho' from data (m)
      startIdx - starting index of segment to be split
        endIdx - ending index of segment to be split
        params - dictionary of parameters

    OUTPUT: alpha - (1D) np array of 'alpha' for each fitted line (rads)
             r - (1D) np array of 'r' for each fitted line (m)
           idx - (N_lines,2) segment's first and last point index
    '''

    ##### TO DO #####
    # Implement a recursive line splitting function
    # It should call 'FitLine()' to fit individual line segments
    # In should call 'FindSplit()' to find an index to split at
    #################
    #initialize the set
    # s1 = np.column_stack( (theta[startIdx:endIdx], rho[startIdx:endIdx]))

    #fit a line to the data points in our current set
    alpha, r = FitLine(theta[startIdx:endIdx],rho[startIdx:endIdx])
    print('start index ', startIdx)
    print('end index ', endIdx)
    #calculate the index of the line to split at
    splitidx = FindSplit(theta[startIdx:endIdx],rho[startIdx:endIdx],alpha,r,params)

    if splitidx == -1: #if it wasn't possible to split, we return that index
        idx = np.array([[startIdx, endIdx]])
        alpha = np.array([alpha])
        r = np.array([r])
    else: # if it is too big, then we need to split
        alpha1, r1, idx1 = SplitLinesRecursive(theta, rho, startIdx, startIdx + splitidx, params)
        alpha2, r2, idx2 = SplitLinesRecursive(theta, rho, startIdx + splitidx, endIdx, params)
        alpha = np.concatenate((alpha1,alpha2))
        r = np.concatenate((r1, r2))
        idx = np.concatenate((idx1,idx2))
    
    return alpha, r, idx


def FindSplit(theta, rho, alpha, r, params):
    '''
        This function takes in a line segment and outputs the best index 
        at which to split the segment

        INPUT:  theta - (1D) np array of angle 'theta' from data (rads)
               rho - (1D) np array of distance 'rho' from data (m)
             alpha - 'alpha' of input line segment (1 number)
                 r - 'r' of input line segment (1 number)
            params - dictionary of parameters

        OUTPUT: SplitIdx - idx at which to split line (return -1 if
                        it cannot be split)
    '''
    ##### TO DO #####
    # Implement a function to find the split index (if one exists)
    # It should compute the distance of each point to the line.
    # The index to split at is the one with the maximum distance
    # value that exceeds 'LINE_POINT_DIST_THRESHOLD', and also does
    # not divide into segments smaller than 'MIN_POINTS_PER_SEGMENT'
    # return -1 if no split is possiple
    #################
    distances = (rho*np.cos(theta-alpha)-r)**2
    idx = np.argmax(distances)
    val = distances[idx]
    #segment length
    seg = float(idx+1)

    #if the distance exceeds this threshold and there are enough points to make a segment
    #we have an index to split and and should return
    if val > params["LINE_POINT_DIST_THRESHOLD"] and seg > params["MIN_POINTS_PER_SEGMENT"]: 
        splitIdx = idx
    else: #not possible to split
        return -1

    return splitIdx

def FitLine(theta, rho):
    '''
    FitLine

    This function outputs a best fit line to a segment of range
    data, expressed in polar form (alpha, r)

    INPUT:  theta - (1D) np array of angle 'theta' from data (rads)
           rho - (1D) np array of distance 'rho' from data (m)

    OUTPUT: alpha - 'alpha' of best fit for range data (1 number) (rads)
             r - 'r' of best fit for range data (1 number) (m)
    '''

    ##### TO DO #####
    # Implement a function to fit a line to polar data points
    # based on the solution to the least squares problem (see Hw)
    #################
    n = float(len(theta))
    print('n ',n)
    val11 = 0.0
    val12 = 0.0
    val22 = 0.0
    val21 = 0.0
    r = 0.0
    for i in range(len(theta)):
        val11 += rho[i]**2 * np.sin(2*theta[i])
        val21 += rho[i]**2 * np.cos(2*theta[i])
        for j in range(len(theta)):
            val12 += rho[i] * rho[j] * np.cos(theta[i]) * np.sin(theta[j])
            val22 += rho[i] * rho[j] * np.cos(theta[i] + theta[j])
    
    x1 = val11 - 2.0/n * val12
    x2 = val21 - 1.0/n * val22

    alpha = 1.0/2.0 * np.arctan2(x1,x2) + np.pi/2.0
    
    for i in range(len(theta)):
        r += rho[i] * np.cos(theta[i]- alpha)   
    r = r * 1.0/n

    # r = 1.0/n * np.sum(rho*np.cos(theta-alpha))
    print('alpha ', alpha, 'r: ', r)
    # plt.plot(theta,rho)
    # plt.show()
    return alpha, r


def MergeColinearNeigbors(theta, rho, alpha, r, pointIdx, params):
    '''
    MergeColinearNeigbors
    This function merges neighboring segments that are colinear and outputs
    a new set of line segments
    INPUT:  theta - (1D) np array of angle 'theta' from data (rads)
              rho - (1D) np array of distance 'rho' from data (m)
            alpha - (1D) np array of 'alpha' for each fitted line (rads)
                r - (1D) np array of 'r' for each fitted line (m)
         pointIdx - (N_lines,2) segment's first and last point indices
           params - dictionary of parameters
    OUTPUT: alphaOut - output 'alpha' of merged lines (rads)
                rOut - output 'r' of merged lines (m)
         pointIdxOut - output start and end indices of merged line segments
    '''
    ##### TO DO #####
    # Implement a function to merge colinear neighboring line segments
    # HINT: loop through line segments and try to fit a line to data
    #       points from two adjacent segments. If this line cannot be
    #       split, then accept the merge. If it can be split, do not merge.
    #################

    # for i in range(len(pointIdx)):
        # alphanew, rnew = FitLine(theta[pointIdx[i]], rho[pointIdx[i]]])
    mergestartidx = pointIdx[0,0]
    mergeendidx = pointIdx[1,1]
    alphatest, rtest = FitLine(theta[mergestartidx:mergeendidx], rho)
    alphaOut = 1
    rOut = 1
    pointIdxOut = 1
    
    return alphaOut, rOut, pointIdxOut


#----------------------------------
# ImportRangeData
def ImportRangeData(filename):

    data = np.genfromtxt('./RangeData/'+filename, delimiter=',')
    x_r = data[0, 0]
    y_r = data[0, 1]
    theta = data[1:, 0]
    rho = data[1:, 1]
    return (x_r, y_r, theta, rho)
#----------------------------------


############################################################
# Main
############################################################
def main():
    # parameters for line extraction (mess with these!)
    MIN_SEG_LENGTH = 0.05  # minimum length of each line segment (m)
    LINE_POINT_DIST_THRESHOLD = 0.02  # max distance of pt from line to split
    MIN_POINTS_PER_SEGMENT = 4  # minimum number of points per line segment
    MAX_P2P_DIST = 1.0  # max distance between two adjent pts within a segment

    # Data files are formated as 'rangeData_<x_r>_<y_r>_N_pts.csv
    # where x_r is the robot's x position
    #       y_r is the robot's y position
    #       N_pts is the number of beams (e.g. 180 -> beams are 2deg apart)

    filename = 'rangeData_5_5_180.csv'
    # filename = 'rangeData_4_9_360.csv'
    # filename = 'rangeData_7_2_90.csv'

    # Import Range Data
    RangeData = ImportRangeData(filename)

    params = {'MIN_SEG_LENGTH': MIN_SEG_LENGTH,
              'LINE_POINT_DIST_THRESHOLD': LINE_POINT_DIST_THRESHOLD,
              'MIN_POINTS_PER_SEGMENT': MIN_POINTS_PER_SEGMENT,
              'MAX_P2P_DIST': MAX_P2P_DIST}

    alpha, r, segend, pointIdx = ExtractLines(RangeData, params)

    ax = PlotScene()
    ax = PlotData(RangeData, ax)
    ax = PlotRays(RangeData, ax)
    ax = PlotLines(segend, ax)

    plt.show(ax)

############################################################

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
