import numpy as np
import cv2

# ============================================================================
FINAL_LINE_COLOR = (66, 244, 98)
WORKING_LINE_COLOR = (244, 65, 95)
ALPHA = 0.25
LINES_THICKNESS = 3

# ============================================================================

class PolygonDrawer(object):
    def __init__(self, window_name,canvas,n_feeder = None):
        self.window_name = window_name # Name for our window
        self.canvas = canvas.copy()
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self.n_feeder = n_feeder
        
        


    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            #print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            #print("Completing polygon with %d points." % len(self.points))
            self.done = True

    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, self.canvas)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            canvas = self.canvas.copy()
            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv2.polylines(canvas, np.array([self.points]), False, FINAL_LINE_COLOR, LINES_THICKNESS)
                # And  also show what the current segment would look like
                cv2.line(canvas, self.points[-1], self.current, WORKING_LINE_COLOR, LINES_THICKNESS)
                
            # Update the window
            height = np.size(canvas, 0)
            width = np.size(canvas, 1)
            cv2.putText(canvas,"Selecione o comedouro "  + str(self.n_feeder),(int(width/2)- 200,int(0.1*height) ),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(self.window_name, canvas)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27: # ESC hit
                self.done = True

        # User finised entering the polygon points, so let's make the final drawing
        canvas = self.canvas.copy()
        
        canvasBlack = np.zeros(self.canvas.shape, np.uint8)
        # of a filled polygon
        if (len(self.points) > 0):
            cv2.fillPoly(canvasBlack, np.array([self.points]), FINAL_LINE_COLOR)
            canvas = cv2.addWeighted(canvasBlack,ALPHA,canvas, 1-ALPHA,10)
            if(self.n_feeder is not None):
                cX,cY = find_center_of_polygon(canvasBlack)
                cv2.putText(canvas,str(self.n_feeder),(cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
        # And show it
        cv2.imshow(self.window_name, canvas)
        # Waiting for the user to press any key
        cv2.waitKey()

        cv2.destroyWindow(self.window_name)
        return canvas

# ============================================================================

def find_center_of_polygon(image):
    #cv2.imshow("mask",image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    #cv2.imshow("mask",thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]
    
    
    # loop over the contours
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        return cX, cY

	
def grab_canvas_from_video(videoPath):
    videoFile = cv2.VideoCapture(videoPath)
    totalFrames = int(videoFile.get(cv2.CAP_PROP_FRAME_COUNT))
    videoFile.set(cv2.CAP_PROP_POS_FRAMES, totalFrames/2)
    return videoFile.read()

def get_feeder_points(videoPath,feeder_n = None):
    success, mCanvas = grab_canvas_from_video(videoPath)
    if(success):
        pd = PolygonDrawer("Feeder", mCanvas,feeder_n)
        image = pd.run()
        return pd.points

    
if __name__ == "__main__":
    videoPath = str(input("Digite o nome do arquivo de video:\n"))
    #print("Pontos do comedouro: %s\n" % get_feeder_points(videoPath))
