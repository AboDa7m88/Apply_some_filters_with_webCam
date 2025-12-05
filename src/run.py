import cv2
import numpy as np
import time

# initial sigma value
sigma = 1.0
gaussian_on = True  # toggle for applying gaussian blur

# function to apply gaussian smoothing
def apply_gaussian_blur(frame_gray, sigma):
    # ensure kernel size is odd and >= 3
    ksize = max(3, int(6 * sigma + 1))
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(frame_gray, (ksize, ksize), sigma)

# function to draw instructions on the frame (with a semi-transparent box)
def draw_overlay(frame, sigma, gaussian_on, mode_name, fps):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # panel rectangle
    panel_w = 300
    panel_h = 250
    x0, y0 = 10, 10
    # draw filled rectangle (semi-transparent)
    alpha = 0.6
    cv2.rectangle(overlay, (x0, y0), (x0+panel_w, y0+panel_h), (0,0,0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # texts
    lines = [
        "Controls (press in video window):",
        "O - Original",
        "X - Sobel X",
        "Y - Sobel Y",
        "M - Sobel magnitude",
        "S - Sobel+Thresholds",
        "L - LoG",
        "'+' - Increase Sigma",
        "'-' - Decrease Sigma",
        "G - Toggle Gaussian (ON/OFF)",
        "Q - Quit",
        f"Mode: {mode_name}",
        f"Sigma: {sigma:.1f}    "
        f"Gaussian: {'ON' if gaussian_on else 'OFF'}",

    ]

    y = y0 + 24
    for i, t in enumerate(lines):
        color = (200, 255, 200) if i < 8 else (200, 200, 255)
        cv2.putText(frame, t, (x0 + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 18

# start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check camera index or permissions.")

current_display = "original"
prev_time = time.time()
fps = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # compute FPS
    cur_time = time.time()
    dt = cur_time - prev_time
    prev_time = cur_time
    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

    # convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # apply gaussian only if enabled
    if gaussian_on:
        blurred_gray = apply_gaussian_blur(gray, sigma)
    else:
        blurred_gray = gray.copy()

    # select what to display
    if current_display == "original":
        if gaussian_on:
            # apply Gaussian to color channels
            b, g, r = cv2.split(frame)
            b = apply_gaussian_blur(b, sigma)
            g = apply_gaussian_blur(g, sigma)
            r = apply_gaussian_blur(r, sigma)
            display = cv2.merge([b, g, r])
        else:
            display = frame.copy()

    elif current_display == "sobel_x":
        sobel_x = cv2.Sobel(blurred_gray, cv2.CV_64F, 1, 0)
        display = cv2.convertScaleAbs(sobel_x)
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    elif current_display == "sobel_y":
        sobel_y = cv2.Sobel(blurred_gray, cv2.CV_64F, 0, 1)
        display = cv2.convertScaleAbs(sobel_y)
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    elif current_display == "magnitude":
        sobel_x = cv2.Sobel(blurred_gray, cv2.CV_64F, 1, 0)
        sobel_y = cv2.Sobel(blurred_gray, cv2.CV_64F, 0, 1)
        magnitude = cv2.magnitude(sobel_x, sobel_y)
        display = cv2.convertScaleAbs(magnitude)
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    elif current_display == "threshold":
        sobel_x = cv2.Sobel(blurred_gray, cv2.CV_64F, 1, 0)
        sobel_y = cv2.Sobel(blurred_gray, cv2.CV_64F, 0, 1)
        mag = cv2.magnitude(sobel_x, sobel_y)
        mag = cv2.convertScaleAbs(mag)
        _, th = cv2.threshold(mag, 50, 255, cv2.THRESH_BINARY)
        display = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

    elif current_display == "log":
        log = cv2.Laplacian(blurred_gray, cv2.CV_64F)
        display = cv2.convertScaleAbs(log)
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    else:
        display = frame.copy()

    # draw overlay (instructions + sigma + mode + fps)
    mode_name = current_display.upper()
    draw_overlay(display, sigma, gaussian_on, mode_name, fps)

    # show window
    cv2.imshow("Output", display)

    # read key
    key = cv2.waitKeyEx(1)

    # letter keys (both lower and upper)
    if key in (ord('o'), ord('O')):
        current_display = "original"
    elif key in (ord('x'), ord('X')):
        current_display = "sobel_x"
    elif key in (ord('y'), ord('Y')):
        current_display = "sobel_y"
    elif key in (ord('m'), ord('M')):
        current_display = "magnitude"
    elif key in (ord('s'), ord('S')):
        current_display = "threshold"
    elif key in (ord('l'), ord('L')):
        current_display = "log"
    elif key in (ord('g'), ord('G')):  # toggle gaussian on/off
        gaussian_on = not gaussian_on

    # plus keys: handle several possible codes (main keyboard, shifted '=' , OEM, numpad)
    elif key in (ord('+'), ord('='), 0xBB, 0x6B):
        sigma = min(10.0, sigma + 0.5)
    # minus keys: main, OEM, numpad
    elif key in (ord('-'), 0xBD, 0x6D):
        sigma = max(0.5, sigma - 0.5)

    elif key in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()
