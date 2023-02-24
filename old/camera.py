# Fooling around with the camera
# import the opencv library
import cv2


def main():
    # define a video capture object
    vid = cv2.VideoCapture(0)

    while True:

        # Capture the video frame by frame
        ret, frame = vid.read()

        # Convert to B&W
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', gray)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
