# ece592hw3

"""
This program has two main functions: detectcircles(filename) and detectstopsign(filename)

**py Homework3.py detectcircles(filename)**
detectcircles(filename) takes as input ColorBlobs.jpg or ColorBlobs5.jpg. 
It uses HSV and Hough circles to find chalk circles. Circles are overlain, the radii are found, and a polygon connects all of the centers (in any order).

Room for improvement: This code is bloated and needs to be cleaned up. I think it would be cool to search for circles and discover the HSV bands within them.

**py Homework3.py detectstopsign(filename)**
detectstopsign(filename) takes as input Stop1.png, Stop2.jpg, Stop3.jpg, Stop4.jpg, or Stop5.jpg.
The program uses two masks for the red stop sign and then uses contour detection to put a box around the stop sign.
The width of one of the eight sides of the stop sign is calculated and added.

Room for improvement: I saw a method that used a moving window, image pyramids, and rootmeansquared in order to deect any stop sign.
That would be more rigorous than the method we were asked to do. We were discouraged from using other people's ideas for this adssignemnt.
"""
