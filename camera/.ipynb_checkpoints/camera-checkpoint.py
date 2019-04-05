{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import cv2\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.image as mpimg\n",
    "\n",
    "class Camera:\n",
    "    def __init__(self, name):\n",
    "        self.name = name\n",
    "    \n",
    "    def find_chessboard_corners(image, nx, ny, ):\n",
    "        # Convert to grayscale\n",
    "        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n",
    "        # Find the chessboard corners\n",
    "        return cv2.findChessboardCorners(gray, (nx, ny), None)\n",
    "    \n",
    "    def find_chessboard_corners_and_draw(image, nx, ny, ):\n",
    "        # Convert to grayscale\n",
    "        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n",
    "        # Find the chessboard corners\n",
    "        # Find the chessboard corners\n",
    "        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)\n",
    "        # If found, draw corners\n",
    "        if ret == True:\n",
    "            # Draw and display the corners\n",
    "            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)\n",
    "        return ret, corners\n",
    "\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
