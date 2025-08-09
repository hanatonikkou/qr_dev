import cv2
import numpy as np
import os

BW_THRESH = 128

def draw_cross(image, point, color):
    cv2.line(image, (point[0] - 4, point[1]), (point[0] + 4, point[1]), color, 2)
    cv2.line(image, (point[0], point[1] - 4), (point[0], point[1] + 4), color, 2)

# fragment is byte-wide data that has a value of either 0 or 255
# returns a tuple of (run color, run length, remaining fragment data)
def color_run(fragment):
    run_color = fragment[0] # current run color is by definition the first color of the fragment
    run_length = 1
    for f in fragment:
        if f == run_color:
            run_length += 1
        else:
            break
    return (run_color, run_length, fragment[run_length:])


# search a line of data for a 1:1:3:1:1 ratio of black:white:black:white:black
# this uses the "state machine" method
# "row_normal" means we're searching by rows
def finder_finder_sm(y, line, row_normal=True):
    # method:
    #  - core primitive is advance-to-next-color. This returns a run length and
    #    color argument
    x = 0
    sequence = [(0,0,0)] * 5
    candidates = []
    avg_width = 0
    while len(line) > 0 :
        (color, run_length, line) = color_run(line)
        x += run_length
        sequence = sequence[1:] + [(run_length, color, x)]
        if sequence[0][0] != 0 and sequence[0][1] == 0: # sequence of 5 and black is in the right position
            # print(sequence)
            ratios = []
            denom = sequence[0][0]
            for seq in sequence:
                ratios.append(int(seq[0] / denom))
            LOWER_1 = 0 # 0.5 ideally, but have to go to 0 for fixed point impl
            UPPER_1 = 2
            LOWER_3 = 2
            UPPER_3 = 4
            if ratios[1] >= LOWER_1 and ratios[1] <= UPPER_1 and ratios[2] >= LOWER_3 and ratios[2] <= UPPER_3 \
                and ratios[3] >= LOWER_1 and ratios[3] <= UPPER_1 and ratios[4] >= LOWER_1 and ratios[4] <= UPPER_1:
                if row_normal:
                    # print(f"{sequence[2][2]},{y} -- {ratios}")
                    candidates.append((sequence[2][2] - sequence[2][0] // 2 - 1, y))
                else:
                    # print(f"{y}, {sequence[2][2]} -- {ratios}")
                    candidates.append((y, sequence[2][2] - sequence[2][0] // 2 - 1))
                for s in sequence:
                    avg_width += s[0]

    return (candidates, int(avg_width / max(len(candidates), 1)))

def read_qr(file):

    if cv2.haveImageReader(file) is False: print("can't open this file"); return
    qr_raw = cv2.imread(file)

    (_r, rgb) = cv2.threshold(qr_raw, BW_THRESH, 255, cv2.THRESH_BINARY)
    binary = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    row_candidates = []
    row_widths = []
    for i, row in enumerate(binary):
        (candidate, width) = finder_finder_sm(i, row, row_normal=True)
        if len(candidate) > 0:
            row_candidates += candidate
            row_widths += [(candidate[0], width)]

    overlay = np.zeros(rgb.shape, np.uint8)
    for candidate in row_candidates:
        cv2.line(overlay, candidate, candidate, (0, 255, 0), 1)
    blended = cv2.addWeighted(overlay, 1.0, rgb, 0.5, 0)

    col_candidates = []
    for i, col in enumerate(binary.T):
        (candidate, _width) = finder_finder_sm(i, col, row_normal=False)
        if len(candidate) > 0:
            col_candidates += candidate

    overlay = np.zeros(rgb.shape, np.uint8)
    for candidate in col_candidates:
        cv2.line(overlay, candidate, candidate, (255, 0, 255), 1)

    blended = cv2.addWeighted(overlay, 1.0, blended, 0.5, 0)

    intersected = np.zeros(rgb.shape, np.uint8)
    marks = []
    finder_width = 0
    for candidate in list(set(row_candidates).intersection(set(col_candidates))):
        # cv2.line(intersected, candidate, candidate, (0, 255, 0), 1)
        for r in row_widths:
            if candidate == r[0]:
                finder_width += r[1]
        marks += [candidate]

    if len(marks) == 3:
        print("candidate good")
        print(f"{marks}")
    else:
        print("no good, this one.")
        return


    d1 = np.sqrt((marks[1][0] - marks[0][0])**2 + (marks[1][1] - marks[0][1])**2)
    d2 = np.sqrt((marks[2][0] - marks[1][0])**2 + (marks[2][1] - marks[1][1])**2)
    d3 = np.sqrt((marks[0][0] - marks[2][0])**2 + (marks[0][1] - marks[2][1])**2)

    if d1 > d2 and d1 > d3: UL_corner = marks[2]; sec_corner = marks[0]; third_corner = marks[1];
    if d2 > d1 and d2 > d3: UL_corner = marks[0]; sec_corner = marks[1]; third_corner = marks[2];
    if d3 > d1 and d3 > d2: UL_corner = marks[1]; sec_corner = marks[2]; third_corner = marks[0];

    for mark in marks: cv2.circle(intersected, mark, 3, (0,255,0), 1)
    draw_cross(intersected, UL_corner, (0,255,0))

    angle_1 = np.arctan2((sec_corner[1] - UL_corner[1]), (sec_corner[0] - UL_corner[0])) *180 /np.pi
    angle_2 = np.arctan2((third_corner[1] - UL_corner[1]), (third_corner[0] - UL_corner[0])) *180 /np.pi
    if angle_1 < 0: angle_1 = 360 + angle_1
    if angle_2 < 0: angle_2 = 360 + angle_2
    difference = angle_1 - angle_2
    if difference < 0: difference = 360 + difference
    if difference < 180: LL_corner = sec_corner; UR_corner = third_corner;
    else: LL_corner = third_corner; UR_corner = sec_corner;
    cv2.line(intersected, UL_corner, UR_corner, (0,255,0), 1)
    blended_intersection = cv2.addWeighted(intersected, 1.0, rgb, 0.5, 0)

    src_pts = np.array([UL_corner, UR_corner, LL_corner]).astype(np.float32)
    w = rgb.shape[1]
    h = rgb.shape[0]
    tgt_pts = np.array([(w*0.15, h*0.15), (w*0.85, h*0.15), (w*0.15, h*0.85)]).astype(np.float32)

    matrix = cv2.getAffineTransform(src_pts, tgt_pts)
    affined = cv2.warpAffine(rgb, matrix, (w, h))

    show = cv2.hconcat([ blended_intersection, affined ])
    print(f"{blended_intersection.shape}")
    print(f"{affined.shape}")
    cv2.imshow('demo', show)
    cv2.waitKey(0)

directory = 'images/'

for file in os.scandir(directory):
    if file.is_file(): read_qr(file.path)
