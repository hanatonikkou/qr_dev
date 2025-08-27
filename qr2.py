import cv2
import numpy as np
import os

BW_THRESH = 128
LO_LIMIT = 1 / 7
UP_LIMIT = 1 / 1.2

def draw_cross(image, point, color):
    cv2.line(image, (point[0] - 4, point[1]), (point[0] + 4, point[1]), color, 2)
    cv2.line(image, (point[0], point[1] - 4), (point[0], point[1] + 4), color, 2)

# fragment is byte-wide data that has a value of either 0 or 255
# returns a tuple of (run color, run length, remaining fragment data)
def color_run(fragment):
    run_color = fragment[0] # current run color is by definition the first color of the fragment
    run_length = 1
    for f in fragment[1:]:
        if f == run_color:
            run_length += 1
        else:
            break
    return (run_color, run_length, fragment[run_length:])

# check ratios between neighbouring runs of black pixels
# deliberately vague to account for wide ratios, noisy data (low res, low quality sensor)
def check_ratio(left, center, right):
    lower_limit = center * LO_LIMIT
    upper_limit = center * UP_LIMIT
    if lower_limit <= left <= upper_limit and lower_limit <= right <= upper_limit: return True
    else: return False

# return candidates for finder patterns
def finder_finder_sm(y, line, img):
    x = 0
    colors = []
    candidates = []
    line_tmp = line
    while len(line_tmp) > 0 :
        (color, run_length, line_tmp) = color_run(line_tmp)
        colors += [(x, color, run_length, x + run_length)]
        x += run_length

    for i in range(0, len(colors)):

        if colors[i][1] == 255: continue
        if colors[i][2] <= 6: continue

        max_d = colors[i][2] * UP_LIMIT
        whitespace = False

        # check for a suitable fragment to the left
        found = False
        n = 1
        while True:
            if i - 2*n < 0: break
            # check distance between the last pixel of the left run and the start of the center run
            if (colors[i][0] - colors[i-2*n][3]) > max_d: break
            if LO_LIMIT <= (colors[i-2*n][2] / colors[i][2]) <= UP_LIMIT:
                found = True;
                left_bound = colors[i-2*n][0]
                break
            n += 1
        if found == False: continue
        # check for whitespace to the left
        counter = 0
        if left_bound - colors[i][2] >= 0:
            for pixel in range(left_bound - colors[i][2], left_bound):
                if line[pixel] == 255: counter += 1
            if (counter / colors[i][2]) >= 0.8: whitespace = True
        elif left_bound == 0: continue
        else:
            for pixel in range(0, left_bound):
                if line[pixel] == 255: counter += 1
            if (counter / left_bound) >= 0.8: whitespace = True


        # check for a suitable fragment to the right
        found = False
        n = 1
        while True:
            if i + 2*n > len(colors) - 1: break
            if (colors[i+2*n][0] - colors[i][3]) > max_d: break
            if LO_LIMIT <= (colors[i+2*n][2] / colors[i][2]) <= UP_LIMIT:
                found = True
                right_bound = colors[i+2*n][3]
                break
            n += 1
        if found == False: continue
        # check for whitespace to the right
        counter = 0
        if right_bound + colors[i][2] <= len(line):
            for pixel in range(right_bound, right_bound + colors[i][2]):
                if line[pixel] == 255: counter += 1
            if (counter / colors[i][2]) >= 0.8: whitespace = True
        elif right_bound == len(line): continue
        else:
            for pixel in range(right_bound, len(line)):
                if line[pixel] == 255: counter += 1
            if (counter / (len(line) - right_bound)) >= 0.8: whitespace = True

        if whitespace: candidates += [[left_bound, right_bound]]
        # else:
        #     if y > 35 and y < 55 and left_bound >150:
        #         print(f"{whitespace, y, left_bound, right_bound}")
        #         print(f"{counter, colors[i][2], (len(line) - right_bound)}")


    # cv2.imshow('', blended)
    # cv2.waitKey(20)
    # input('')
    return (candidates)

def has_neighbors (candidate, line):
    if line == []: return False
    for i in range(0, len(line)):
        comparator = line[i]
        if candidate[0] <= comparator[0] <= candidate[1] or candidate[0] <= comparator[1] <= candidate[1] \
                or (candidate[0] > comparator[0] and candidate[1] < comparator[1]):
            overlap_upper = candidate[1] if candidate[1] < comparator[1] else comparator[1]
            overlap_lower = candidate[0] if candidate[0] > comparator[0] else comparator[0]
            candidate_length = candidate[1] - candidate[0]
            overlap_length = overlap_upper - overlap_lower
            if overlap_length / candidate_length > 0.8: return (True, i)
            else: continue
        else: continue
    return False

# keep only groups of overlapping candidates
def reject_candidates (candidates):
    length = len(candidates)
    area_no = 0

    # reject candidates with not enough neighbours outright but allow some empty rows
    for line_n in range(0, length):
        if candidates[line_n]:
            line_n_length = len(candidates[line_n])
            for index in range(0, line_n_length):
                keep_or_not = candidates[line_n][line_n_length - index - 1]
                plus1 = False
                plus2 = False
                plus3 = False
                plus4 = False
                if line_n < length - 2:
                    plus1 = has_neighbors(keep_or_not, candidates[line_n+1])
                if line_n < length - 3:
                    plus2 = has_neighbors(keep_or_not, candidates[line_n+2])
                if line_n < length - 4:
                    plus3 = has_neighbors(keep_or_not, candidates[line_n+3])
                if line_n < length - 5:
                    plus4 = has_neighbors(keep_or_not, candidates[line_n+4])
                    pass

                counter = 0
                if index >= length - 5: counter += 1 # make algorithm work at the edge of the image
                if index >= length - 4: counter += 1
                if index >= length - 3: counter += 1
                if index >= length - 2: counter += 1
                if plus1: counter += 1
                if plus2: counter += 1
                if plus3: counter += 1
                if plus4: counter += 1
                # if len(keep_or_not) == 2 and counter < 2: candidates[line_n].remove(keep_or_not)

                # label groups of overlapping candidates
                if len(keep_or_not) == 2:
                    if counter < 2:
                        candidates[line_n].remove(keep_or_not)
                        continue
                    else:
                        keep_or_not += [area_no]
                        area_no += 1
                if plus1 and len(candidates[line_n+1][plus1[1]]) == 2:
                    candidates[line_n+1][plus1[1]] += [keep_or_not[2]]
                if plus2 and len(candidates[line_n+2][plus2[1]]) == 2:
                    candidates[line_n+2][plus2[1]] += [keep_or_not[2]]
                if plus3 and len(candidates[line_n+3][plus3[1]]) == 2:
                    candidates[line_n+3][plus3[1]] += [keep_or_not[2]]
                if plus4 and len(candidates[line_n+4][plus4[1]]) == 2:
                    candidates[line_n+4][plus4[1]] += [keep_or_not[2]]

    # count lines in each group, mark corner points for each group
    # keep only groups with the highest line count
    # remove groups with too large a ratio between first and last one
    #   (these are loose groupings of false positives)
    #
    # data structure:
    #   index
    #   counter of lines in the group
    #   first line of group: row / col - beginning - end
    #   last line of group: row / col - beginning - end
    counter = np.full(area_no, 0)
    candidate_groups = [[0, 0, [], []]]
    for i in range(1, area_no):
        candidate_groups += [[i, 0, [], []]]

    for index, line_n in enumerate(candidates):
        for candidate in line_n:
            group_no = candidate[2]
            candidate_groups[group_no][1] += 1
            counter[group_no] += 1
            if candidate_groups[group_no][1] == 1: candidate_groups[group_no][2] = [index, candidate[:2]]
            else: candidate_groups[group_no][3] = [index, candidate[:2]]

    # sometimes a line will be grouped by itself, erroneously
    for each in reversed(candidate_groups):
        if each[1] == 1: candidate_groups.remove(each)

    if area_no > 5:
        keep_max = np.sort(counter)[area_no - 5]
        for each in candidate_groups:
            length1 = each[2][1][1] - each[2][1][0]
            length2 = each[3][1][1] - each[3][1][0]
            if each[1] < keep_max: candidate_groups.remove(each)
            elif (length1 / length2) < 0.7 or (length2 / length1) < 0.7:
                candidate_groups.remove(each)

    # for each in candidates:
    #     if each: print(f"{each}")
    # candidates:
    #   per line: [start, finish, group_no]
    #   e.g. [[96, 170, 27][124, 210, 29]]
    #
    # candidate_groups:
    #    [group_no, line count, [first line [start, finish]], [last line [start, finish]]]
    #    e.g. [1, 18. [36, [29, 70]], [53, [29, 69]]]
    #
    # return_candidates:
    #   [group_no, [[line, [start, finish]], [line, [start, finish]], ... ]]
    #   e.g. [5, [[90, [61. 116]], [91, [63, 115]], ... ]]


    if len(candidate_groups) < 3: print(f"not enough candidates"); return False
    return_candidates = [[]]
    for i, group in enumerate(candidate_groups):
        # print(f"{each}")
        group_no = group[0]
        no_of_lines = group[1]
        line = group[2][0]
        n = 0

        if i == len(return_candidates): return_candidates += [[]]
        return_candidates[i]= [i, []]
        while n <= no_of_lines:
            for each in candidates[line]:
                if each[2] == group_no:
                    return_candidates[i][1] += [[line, each[:2]]]
                    line +=1
                    # print(f"{return_candidates}")
            n += 1
    # for each in return_candidates:
    #     for el in each[1]:
    #         print(f"{el}")
    # for each in return_candidates:
    #     print(f"{each}")

    # return candidate_groups
    return return_candidates

def read_qr(file):

    if cv2.haveImageReader(file) is False: print("can't open this file"); return
    qr_raw = cv2.imread(file)

    (_r, rgb) = cv2.threshold(qr_raw, BW_THRESH, 255, cv2.THRESH_BINARY)
    binary = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    row_candidates = [[]]
    for i, row in enumerate(binary):
        row_candidates[i] = finder_finder_sm(i, row, rgb)
        row_candidates += [[]]
    row_candidates = reject_candidates(row_candidates)
    if row_candidates == False: return
    # for row in row_candidates:
    #     if row: print(f"{row}")

    col_candidates = [[]]
    for i, col in enumerate(binary.T):
        col_candidates[i] = finder_finder_sm(i, col, rgb)
        col_candidates += [[]]
    col_candidates = reject_candidates(col_candidates)
    if col_candidates == False: return

    # overlay = np.zeros(rgb.shape, np.uint8)
    # for group in row_candidates:
    #     for each in group[1]:
    # for group in col_candidates:
    #     for each in group[1]:
    #         cv2.line(overlay, (each[0], each[1][0]), (each[0], each[1][1]), (0,255,0), 1)

    #
    # for testing the return of finder_finder_sm
    # # for i, col in enumerate(col_candidates):
    # #     if col:
    # #         for candidate in col:
    # #             cv2.line(overlay, (i, candidate[0]), (i, candidate[1]), (255, 0, 255), 1)

    # count overlapping pixels
    pixel_counts = np.zeros((len(row_candidates), len(col_candidates)), np.uint32)
    # unpack each group of row candidates
    for r_index, r_group in enumerate(row_candidates):
        r_group_no = r_group[0]
        r_candidates = r_group[1]
        # unpack each [row, [start, end]]
        for r_can in r_candidates:
            r_line = r_can[0]
            r_start = r_can[1][0]
            r_end = r_can[1][1]
            # unpack each group of column candidates
            for c_group in col_candidates:
                c_group_no = c_group[0]
                c_candidates = c_group[1]
                counter = 0
                # unpack each [column, [start, end]]
                for c_can in c_candidates:
                    c_line = c_can[0]
                    c_start = c_can[1][0]
                    c_end = c_can[1][1]
                    # print(f"r {r_group_no, r_line} c {c_group_no, c_can}")
                    if (r_start <= c_line <= r_end) and (c_start <= r_line <= c_end):
                        counter += 1
                        # increment the counter for the respective row / col groups
                        pixel_counts[r_group_no][c_group_no] += 1
                # if counter != 0: print(f"{r_group_no, c_group_no, counter}")
    max_pixels = []
    # (pixel count, group row, group col)
    for i, group in enumerate(pixel_counts):
        max_pixels += [(group[np.argmax(group)], i, np.argmax(group))]
    max_pixels = sorted(max_pixels)[len(max_pixels)-3:]

    marks = []
    for each in max_pixels[:3]:
        # print(f"{each}")
        for group in row_candidates:
            if group[0] == each[1]:
                left = 0
                right = 0
                counter = 0
                for el in group[1]:
                    left += el[1][0]
                    right += el[1][1]
                    counter += 1
                    if counter < 3: print(f"{el}")
                left = round(left / counter)
                right = round(right / counter)
                col = round((left + right) / 2)
                # print(f"{left, right, col}")
        for group in col_candidates:
            if group[0] == each[2]:
                up = 0
                down = 0
                counter = 0
                for el in group[1]:
                    up += el[1][0]
                    down += el[1][1]
                    counter += 1
                    if counter < 3: print(f"{el}")
                up = round(up / counter)
                down = round(down / counter)
                row = round((up + down) / 2)
                # print(f"{up, down, row}")
        print(f"r {row} c {col}")
        # cv2.circle(overlay, (col, row), 5, (0,255,0), 1)
        marks += [(col, row)]





    if len(marks) == 3:
        print("candidate good")
        print(f"{marks}")
    else:
        overlay = np.zeros(rgb.shape, np.uint8)
        for mark in marks: cv2.circle(overlay, mark, 3, (0,255,0), 1)
        blended = cv2.addWeighted(overlay, 1.0, rgb, 0.5, 0)
        cv2.imshow('image', blended)
        cv2.waitKey(0)
        print("no good, this one.")
        return

    d1 = np.sqrt((marks[1][0] - marks[0][0])**2 + (marks[1][1] - marks[0][1])**2)
    d2 = np.sqrt((marks[2][0] - marks[1][0])**2 + (marks[2][1] - marks[1][1])**2)
    d3 = np.sqrt((marks[0][0] - marks[2][0])**2 + (marks[0][1] - marks[2][1])**2)

    if d1 > d2 and d1 > d3: UL_corner = marks[2]; sec_corner = marks[0]; third_corner = marks[1];
    if d2 > d1 and d2 > d3: UL_corner = marks[0]; sec_corner = marks[1]; third_corner = marks[2];
    if d3 > d1 and d3 > d2: UL_corner = marks[1]; sec_corner = marks[2]; third_corner = marks[0];

    overlay = np.zeros(rgb.shape, np.uint8)
    for mark in marks: cv2.circle(overlay, mark, 3, (0,255,0), 1)
    draw_cross(overlay, UL_corner, (0,255,0))

    angle_1 = np.arctan2((sec_corner[1] - UL_corner[1]), (sec_corner[0] - UL_corner[0])) *180 /np.pi
    angle_2 = np.arctan2((third_corner[1] - UL_corner[1]), (third_corner[0] - UL_corner[0])) *180 /np.pi
    if angle_1 < 0: angle_1 = 360 + angle_1
    if angle_2 < 0: angle_2 = 360 + angle_2
    difference = angle_1 - angle_2
    if difference < 0: difference = 360 + difference
    if difference < 180: LL_corner = sec_corner; UR_corner = third_corner;
    else: LL_corner = third_corner; UR_corner = sec_corner;
    cv2.line(overlay, UL_corner, UR_corner, (0,255,0), 1)

    src_pts = np.array([UL_corner, UR_corner, LL_corner]).astype(np.float32)
    w = rgb.shape[1]
    h = rgb.shape[0]
    tgt_pts = np.array([(w*0.15, h*0.15), (w*0.85, h*0.15), (w*0.15, h*0.85)]).astype(np.float32)

    matrix = cv2.getAffineTransform(src_pts, tgt_pts)
    affined = cv2.warpAffine(rgb, matrix, (w, h))

    blended = cv2.addWeighted(overlay, 1.0, rgb, 0.5, 0)
    show = cv2.hconcat([ blended, affined ])
    cv2.imshow('demo', show)
    cv2.waitKey(0)

directory = 'images/'

for file in os.scandir(directory):
    if file.is_file(): read_qr(file.path)
