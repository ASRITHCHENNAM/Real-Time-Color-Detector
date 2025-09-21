import cv2
import numpy as np
import pandas as pd
from skimage import color

# ---------- Load Colors CSV ----------
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('colors.csv', names=index, header=None)

# Convert CSV colors to LAB for advanced matching
csv['LAB'] = csv.apply(
    lambda row: color.rgb2lab(
        np.uint8([[[row['R'], row['G'], row['B']]]]) / 255.0
    )[0][0],
    axis=1
)

# ---------- Globals ----------
xpos = ypos = 0
r = g = b = 0

# ---------- Advanced Color Matching ----------
def getColorName(R, G, B):
    clicked_lab = color.rgb2lab(np.uint8([[[R, G, B]]]) / 255.0)[0][0]

    min_dist = float('inf')
    cname = None
    for i in range(len(csv)):
        dist = color.deltaE_ciede2000(clicked_lab, csv.loc[i, 'LAB'])
        if dist < min_dist:
            min_dist = dist
            cname = csv.loc[i, "color_name"]
    return cname

# ---------- Mouse Callback ----------
def mouse_move(event, x, y, flags, param):
    global xpos, ypos, r, g, b, frame
    if event == cv2.EVENT_MOUSEMOVE:
        xpos, ypos = x, y
        b, g, r = frame[y, x]
        b, g, r = int(b), int(g), int(r)

# ---------- Webcam Mode ----------
cap = cv2.VideoCapture(0)
cv2.namedWindow('Color Detector')
cv2.setMouseCallback('Color Detector', mouse_move)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw rectangle at top
    cv2.rectangle(frame, (20, 20), (750, 60), (b, g, r), -1)

    # Display text with color name + RGB + HEX
    hex_val = '#{:02x}{:02x}{:02x}'.format(r, g, b).upper()
    text = f"{getColorName(r, g, b)}  R={r} G={g} B={b}  {hex_val}"

    # Choose text color (black/white) based on brightness
    if r + g + b >= 600:
        cv2.putText(frame, text, (50, 50), 2, 0.8,
                    (0, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, text, (50, 50), 2, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

    # (Removed the cv2.circle line here ✅)

    cv2.imshow("Color Detector", frame)

    # Break loop on ESC
    if cv2.waitKey(20) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()