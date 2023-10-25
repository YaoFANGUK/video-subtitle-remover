import cv2

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Point coordinates ({}, {})".format(x, y))
img = cv2.imread("./example/remove-anything/dog.jpg")

cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click_event)
cv2.waitKey(0)

cv2.destroyAllWindows()
