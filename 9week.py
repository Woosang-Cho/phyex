import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


image = cv2.imread("points.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

big_points_px = []
min_area_threshold = 2000

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area >= min_area_threshold:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            big_points_px.append((cx, cy))
            cv2.circle(image, (cx, cy), 35, (0, 255, 0), 3)

img_height, img_width = image.shape[:2]
cm_per_pixel_x = 21.0 / img_width
cm_per_pixel_y = 29.7 / img_height

big_points_cm = []
print("\n A4 아래 기준으로 점 위치 (cm):")
for (cx, cy) in big_points_px:
    x_cm = cx * cm_per_pixel_x
    y_cm = cy * cm_per_pixel_y
    height_from_bottom = 29.7 - y_cm
    print(f"- 점 ({cx}, {cy}) -> x: {x_cm:.2f} cm, 아래서부터: {height_from_bottom:.2f} cm")
    big_points_cm.append((x_cm, y_cm)) 

cv2.imwrite("points_big_detected.jpg", image)

def remove_outliers(points, threshold=1.5):
    x_mean = np.mean([p[0] for p in points])
    y_mean = np.mean([p[1] for p in points])
    dists = [np.sqrt((x - x_mean)**2 + (y - y_mean)**2) for (x, y) in points]
    median = np.median(dists)
    filtered = [p for p, d in zip(points, dists) if d <= median * threshold]
    return filtered

### 최소 외접원 계산 함수 ###
def dist(center, point):
    return np.sqrt((center[0] - point[0])**2 + (center[1] - point[1])**2)

def max_dist_to_center(center, points):
    distances = [dist(center, point) for point in points]
    return max(distances)

def objective_function(center_radius, points):
    center = (center_radius[0], center_radius[1])
    radius = center_radius[2]
    max_distance = max_dist_to_center(center, points)
    return radius if max_distance <= radius else 1e9

def find_min_circle(points):
    max_distance = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            max_distance = max(max_distance, dist(points[i], points[j]))
    initial_center = (
        (np.min([p[0] for p in points]) + np.max([p[0] for p in points])) / 2,
        (np.min([p[1] for p in points]) + np.max([p[1] for p in points])) / 2,
    )
    initial_radius = max_distance / 2
    initial_guess = (*initial_center, initial_radius)
    bounds = ((None, None), (None, None), (0.1, None))
    result = minimize(objective_function, initial_guess, args=(points,), bounds=bounds, method='Powell')
    return (result.x[0], result.x[1]), result.x[2]

### 외접원 시각화 ###
def draw_circle(points, threshold=1.5):
    filtered_points = remove_outliers(points, threshold)
    center, radius = find_min_circle(filtered_points)

    fig, ax = plt.subplots(figsize=(10, 8))  

    x_all, y_all = zip(*points)
    ax.scatter(x_all, y_all, color='gray', marker='x', s=80, label='All Points')

    x_f, y_f = zip(*filtered_points)
    ax.scatter(x_f, y_f, color='red', marker='x', s=80, label='Used Points')

    ax.scatter(center[0], center[1], color='green', s=100, label='Center')
    circle = plt.Circle(center, radius, color='green', fill=False, linewidth=2, label=f'Min Circle (r={radius:.2f} cm)')
    ax.add_artist(circle)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_title("Minimum Enclosing Circle (after outlier removal)")
    ax.grid(True)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_xticks(np.arange(np.floor(x_min), np.ceil(x_max) + 0.01, 0.5))
    ax.set_yticks(np.arange(np.floor(y_min), np.ceil(y_max) + 0.01, 0.5))
    ax.margins(0.2)

    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    y_from_bottom = [(x, 29.7 - y) for (x, y) in points] 
    y_from_bottom_sorted = sorted(y_from_bottom, key=lambda p: -p[1])

    table_data = [["X (cm)", "Y (cm)"]] 
    for x, h in y_from_bottom_sorted:
        table_data.append([f"{x:.2f}", f"{h:.2f}"])

    table = plt.table(
        cellText=table_data,
        colWidths=[0.1, 0.2], 
        loc='right',
        bbox=[1.02, 0.15, 0.25, 0.5]  
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)  

    plt.subplots_adjust(right=0.75) 
    plt.tight_layout()
    plt.show()


draw_circle(big_points_cm, threshold=1.5)
