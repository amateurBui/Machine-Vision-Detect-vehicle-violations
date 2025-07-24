import cv2
import numpy as np
import math
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, PhotoImage
from PIL import Image, ImageTk
import threading

# Biến trạng thái để kiểm soát việc phát/tạm dừng video
is_paused = False

# Hàm chạy đoạn code phát hiện phương tiện vượt đèn đỏ
def run_detection(pause_callback, play_callback, video_file):
    global is_paused

    # Đọc video dựa trên lựa chọn
    cap = cv2.VideoCapture(video_file)

    # Load mô hình YOLOv10
    model = YOLO('yolov10s.pt')  # Thay 'yolov10s.pt' bằng đường dẫn file trọng số của bạn

    # Danh sách các lớp cần phân loại
    classes_of_interest = ['bicycle', 'motorcycle', 'car', 'truck']

    # Định nghĩa các vùng (các đa giác được xác định bởi tọa độ) dựa trên video
    if video_file == 'MVI_0018.MP4':
        # Làn bên trái (cho bicycle, motorcycle)
        area1 = [(171, 745), (663, 252), (1270, 352), (1203, 748)]
        # Làn bên phải (cho car, truck)
        area2 = [(164, 745), (663, 252), (200, 168), (0, 277)]
        # Vạch dừng đèn đỏ (Line1)
        line1 = [(35, 270), (1154, 563)]
        # Vùng kiểm tra đèn giao thông
        area_traffic_light = [(874, 2), (833, 50), (950, 50), (950, 2)]
    elif video_file == 'MVI_0016.MP4':
        area1 = [(171, 745), (521, 290), (1270, 352), (1203, 748)]
        area2 = [(164, 745), (521, 290), (200, 269), (0, 323)]
        line1 = [(0, 340), (1154, 478)]
        area_traffic_light = [(654, 12), (654, 55), (760, 55), (760, 12)]
    elif video_file == 'MVI_0019.MP4':
        area1 = [(135, 745), (292, 210), (114, 193), (0, 386)]
        area2 = [(164, 745), (299, 260), (856, 201), (1203, 428)]
        line1 = [(949, 222), (1020, 256)]
        area_traffic_light = [(674, 2), (633, 50), (750, 50), (750, 2)]

    # Quy định phương tiện được phép trong từng vùng
    allowed_vehicles = {
        'area1': ['bicycle', 'motorcycle'],
        'area2': ['car', 'truck']
    }

    # Từ điển để theo dõi trạng thái ROI_in của các phương tiện
    roi_status = {}  # key: (label, id), value: ROI_in
    vehicle_ids = {}  # key: (label, cx, cy), value: id
    next_id = 0  # ID tăng dần cho mỗi phương tiện mới

    def point_in_polygon(point, polygon):
        """Kiểm tra xem điểm có nằm trong đa giác không"""
        x, y = point
        polygon = np.array(polygon, np.int32)
        return cv2.pointPolygonTest(polygon, (x, y), False) >= 0

    def is_point_cross_line(point, line_start, line_end):
        """Kiểm tra xem điểm có cắt qua đường thẳng không"""
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end

        # Tính phương trình đường thẳng: y = mx + b
        if x2 - x1 == 0:  # Đường thẳng dọc
            return abs(x - x1) < 5  # Kiểm tra khoảng cách nhỏ hơn ngưỡng
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        # Tính y dự đoán tại x của điểm
        y_pred = m * x + b
        # Kiểm tra điểm nằm phía trên hay dưới đường thẳng
        return y <= y_pred  # Điểm vượt qua nếu nằm phía trên đường thẳng

    def get_vehicle_id(label, cx, cy):
        """Gán ID cho phương tiện dựa trên tọa độ gần nhất"""
        nonlocal next_id
        key = (label, cx, cy)
        
        # Nếu phương tiện đã được gán ID trước đó
        if key in vehicle_ids:
            return vehicle_ids[key]
        
        # Tìm phương tiện cùng loại gần nhất trong frame trước
        min_dist = float('inf')
        closest_id = None
        for (prev_label, prev_cx, prev_cy), vid in vehicle_ids.items():
            if prev_label == label:
                dist = math.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                if dist < min_dist and dist < 50:  # Ngưỡng khoảng cách
                    min_dist = dist
                    closest_id = vid
        
        if closest_id is not None:
            vehicle_ids[key] = closest_id
            return closest_id
        
        # Nếu không tìm thấy, gán ID mới
        vehicle_ids[key] = next_id
        next_id += 1
        return vehicle_ids[key]

    def process_frame(frame):
        # Resize frame để xử lý nhanh hơn
        frame = cv2.resize(frame, (1020, 600))

        # --- Phần xử lý tín hiệu đèn giao thông ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Định nghĩa phạm vi màu
        lower_green = np.array([70, 200, 153])  # Phạm vi màu xanh lá
        upper_green = np.array([90, 255, 255])
        lower_red = np.array([140, 175, 0])    # Phạm vi màu đỏ
        upper_red = np.array([179, 204, 255])

        # Tạo mask cho màu xanh lá và màu đỏ
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        red_mask = cv2.inRange(hsv, lower_red, upper_red)

        # Kết hợp hai mask
        combined_mask = cv2.bitwise_or(green_mask, red_mask)

        # Áp dụng threshold (ảnh binary)
        _, final_mask = cv2.threshold(combined_mask, 254, 255, cv2.THRESH_BINARY)

        # Tạo mask cho vùng kiểm tra đèn giao thông
        traffic_light_mask = np.zeros_like(final_mask)
        traffic_light_points = np.array(area_traffic_light, np.int32)
        cv2.fillPoly(traffic_light_mask, [traffic_light_points], 255)

        # Chỉ giữ lại các vùng nằm trong area_traffic_light (ảnh binary kết hợp vs các ngưỡng từ combined_mask để lọc bớt các tín hiệu nhiễu, chỉ còn mỗi đèn giao thông để detect)
        final_mask = cv2.bitwise_and(final_mask, traffic_light_mask)

        # Tìm contours của đèn giao thông
        cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_traffic_light = None

        for c in cnts:
            if cv2.contourArea(c) < 5:
                continue

            # Đơn giản và xấp xỉ các đường viền
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)

            area_contour = cv2.contourArea(c)
            ((x, y), radius) = cv2.minEnclosingCircle(c) # Tìm vòng tròn nhỏ nhất bao quanh đường viền, trả về tâm và bán kính của vòng tròn.
            circle_area = math.pi * radius * radius # Diện tích của vòng tròn nhỏ nhất
            circularity = area_contour / circle_area if circle_area > 0 else 0 # Đánh giá độ tròn của đường viền, tính bằng tỷ lệ diện tích của đường viền so với diện tích vòng tròn nhỏ nhất bao quanh nó

            # Ngưỡng xét độ tròn của đèn giao thông
            if 0.1 < circularity < 1.3:
                x, y, w, h = cv2.boundingRect(c) # Lấy các hộp bao gán vào 4 biến mang đi xử lý
                cx = x + w // 2
                cy = y + h // 2

                # kiểm tra xem có bao nhiêu pixel khác 0 (màu trắng) trong vùng ROI của green mask
                if cv2.countNonZero(green_mask[y:y+h, x:x+w]) > 2:
                    color = (0, 255, 0)  # Màu xanh lá
                    text_color = (0, 255, 0)
                    label = "GREEN"
                elif cv2.countNonZero(red_mask[y:y+h, x:x+w]) > 2:
                    color = (0, 0, 255)  # Màu đỏ
                    text_color = (0, 0, 255)
                    label = "RED"
                else:
                    continue

                detected_traffic_light = label

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.circle(frame, (cx, cy), 1, (255, 0, 0), -1)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        # --- Phần xử lý phương tiện giao thông (YOLOv10) ---
        results = model.predict(frame, conf=0.5)  # Ngưỡng confidence là 0.5

        # Tạo mask cho các vùng
        area1_mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
        area2_mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
        cv2.fillPoly(area1_mask, [np.array(area1, np.int32)], 255)
        cv2.fillPoly(area2_mask, [np.array(area2, np.int32)], 255)

        ## Lưu loại phương tiện và trạng thái vi phạm
        detected_objects = []
        violations = []

        ## Các kết quả từ YOLO trả về, đem đi xử lý 
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Tọa độ hộp bao
            confidences = result.boxes.conf.cpu().numpy()  # Độ tin cậy
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # ID lớp

            ## lặp qua tất cả các đối tượng được phát hiện trong ảnh và truy xuất các thông tin liên quan đến chúng (hộp bao, độ tin cậy, và ID lớp)
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                label = model.names[cls_id]  # Tên lớp

                if label in classes_of_interest:
                    x1, y1, x2, y2 = map(int, box)  # Tọa độ hộp bao

                    # Tính trung tâm của hộp bao
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cv2.circle(frame, (cx, cy), 1, (255, 0, 0), -1)  # Vẽ điểm trung tâm

                    # Gán ID cho phương tiện
                    vehicle_id = get_vehicle_id(label, cx, cy)
                    key = (label, vehicle_id)

                    # Xác định màu sắc dựa trên lớp
                    if label == 'bicycle':
                        color = (0, 255, 0)  # Màu xanh lá
                    elif label == 'motorcycle':
                        color = (0, 0, 255)  # Màu đỏ
                    elif label == 'car':
                        color = (255, 255, 0)  # Màu vàng
                    elif label == 'truck':
                        color = (255, 0, 255)  # Màu tím

                    detected_objects.append(label)

                    # Kiểm tra trạng thái ROI_in
                    point = (cx, cy)
                    area_name = None
                    if point_in_polygon(point, area1):
                        area_name = 'area1'
                    elif point_in_polygon(point, area2):
                        area_name = 'area2'

                    # Nếu nằm trong ROI
                    if area_name:
                        # Nếu phương tiện đi vào ROI
                        if key not in roi_status:
                            roi_status[key] = 1  # Set ROI_in = 1 khi đi vào ROI
                        # Nếu trong ROI và vượt qua Line1, set ROI_in = 2
                        if is_point_cross_line(point, line1[0], line1[1]):
                            roi_status[key] = 2

                    # Kiểm tra vi phạm làn đường (chỉ trong ROI)
                    if area_name and label not in allowed_vehicles[area_name]:
                        violations.append(f"{label} in {area_name} - WRONG LANE!")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                        cv2.putText(frame, 'WRONG LANE!', (x1, y2 + 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Kiểm tra vi phạm vượt đèn đỏ cho các phương tiện với ROI_in = 2
                    if key in roi_status and roi_status[key] == 2 and area_name:
                        if detected_traffic_light == "RED" and is_point_cross_line(point, line1[0], line1[1]):
                            violations.append(f"{label} (ID:{vehicle_id}) - RUN RED LIGHT!")
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                            cv2.putText(frame, 'RUN RED LIGHT!', (x1, y2 + 20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # In trạng thái
                    cross_line = is_point_cross_line(point, line1[0], line1[1])
                    print(f"Label: {label}, ID: {vehicle_id}, ROI_in: {roi_status.get(key, 0)}, Traffic Light: {detected_traffic_light}, Cross Line: {cross_line}")

        # Vẽ vạch dừng đèn đỏ (Line1)
        #cv2.line(frame, line1[0], line1[1], (0, 0, 255), 2)  # Vẽ đường màu đỏ

        # Vẽ các vùng lên frame
        #cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)  # Area1: Màu xanh dương
        #cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 0), 2)  # Area2: Màu xanh lá
        #cv2.polylines(frame, [np.array(area_traffic_light, np.int32)], True, (0, 255, 255), 2)  # Area traffic light: Màu vàng

        return frame, detected_objects, violations, detected_traffic_light

    count = 0
    last_frame = None  # Lưu frame cuối cùng để hiển thị khi tạm dừng

    while True:
        if not is_paused:
            ret, frame = cap.read()
            count += 1
            if count % 5 != 0:  # Bỏ qua 2 frame để tăng tốc
                continue
            if not ret:
                break

            # Xử lý frame
            processed_frame, detected_objects, violations, detected_traffic_light = process_frame(frame)
            last_frame = processed_frame  # Lưu frame cuối cùng

            # Hiển thị frame
            cv2.imshow("Vehicle and Traffic Light Detection", processed_frame)

            # In danh sách các đối tượng được phát hiện, vi phạm và tín hiệu đèn
            print(f"Frame {count}: Detected objects: {detected_objects}")
            if violations:
                print(f"Violations: {violations}")
            if detected_traffic_light:
                print(f"Traffic Light: {detected_traffic_light}")

        else:
            # Khi tạm dừng, hiển thị frame cuối cùng (nếu có)
            if last_frame is not None:
                cv2.imshow("Vehicle and Traffic Light Detection", last_frame)

        # Kiểm tra phím ESC để thoát
        key = cv2.waitKey(5 if not is_paused else 100)  # Chờ lâu hơn khi tạm dừng để giảm CPU usage
        if key & 0xFF == 27:  # Nhấn 'ESC' để thoát
            break

    cap.release()
    cv2.destroyAllWindows()

# Tạo giao diện GUI
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Violation Detection System")
        self.root.geometry("800x750")  # Tăng chiều cao để chứa radio button

        # Slide 1: Hình ảnh, 6 dòng text, 1 nút nhấn, và radio button
        self.slide1 = tk.Frame(self.root)
        self.slide1.pack(fill="both", expand=True)

        # Hình ảnh (giả định có file 'logo.jpg', bạn cần thay đường dẫn nếu cần)
        try:
            img = Image.open("logo.jpg")  # Thay 'logo.jpg' bằng đường dẫn đến hình ảnh của bạn
            img = img.resize((300, 300), Image.Resampling.LANCZOS)
            self.logo = ImageTk.PhotoImage(img)
            tk.Label(self.slide1, image=self.logo).pack(pady=20)
        except Exception as e:
            tk.Label(self.slide1, text="Không thể tải hình ảnh!").pack(pady=20)

        # 6 dòng text
        tk.Label(self.slide1, text="Xử lý ảnh trong giao thông phân tích video giao thông để phát hiện\n vi phạm và phân tích mật độ người tham gia giao thông để dự đoán tắt đường", font=("Arial", 16, "bold")).pack(pady=5)
        tk.Label(self.slide1, text="Ho Chi Minh City University of Technology and Education – HCMUTE").pack(pady=5)
        tk.Label(self.slide1, text="Student 1: Nguyễn Trung Nhân                    Mssv: 21146033").pack(pady=5)
        tk.Label(self.slide1, text="Student 2: Bùi Minh Thắng                       Mssv: 21146414").pack(pady=5)
        tk.Label(self.slide1, text="Student 3: Võ Trần Quốc Việt                    Mssv: 22146064").pack(pady=5)
        tk.Label(self.slide1, text="Lecturer: PhD. Nguyễn Văn Thái").pack(pady=5)

        # Thêm radio button để chọn video
        tk.Label(self.slide1, text="Select Video for Detection:", font=("Arial", 12)).pack(pady=10)
        self.selected_video = tk.StringVar(value="MVI_0018.MP4")  # Giá trị mặc định
        video_frame = tk.Frame(self.slide1)
        video_frame.pack(pady=5)
        tk.Radiobutton(video_frame, text="MVI_0018.MP4", value="MVI_0018.MP4", variable=self.selected_video, font=("Arial", 10)).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(video_frame, text="MVI_0016.MP4", value="MVI_0016.MP4", variable=self.selected_video, font=("Arial", 10)).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(video_frame, text="MVI_0019.MP4", value="MVI_0019.MP4", variable=self.selected_video, font=("Arial", 10)).pack(side=tk.LEFT, padx=10)

        # Nút nhấn chuyển sang Slide 2
        tk.Button(self.slide1, text="Start Detection", command=self.show_slide2, font=("Arial", 12), bg="green", fg="white").pack(pady=20)

        # Slide 2: Chạy video detection
        self.slide2 = tk.Frame(self.root)

    def pause_video(self):
        global is_paused
        is_paused = True

    def play_video(self):
        global is_paused
        is_paused = False

    def show_slide2(self):
        # Ẩn Slide 1 và hiển thị Slide 2
        self.slide1.pack_forget()
        self.slide2.pack(fill="both", expand=True)

        # Thêm nhãn để báo đang chạy
        tk.Label(self.slide2, text="Running Traffic Violation Detection...", font=("Arial", 16, "bold")).pack(pady=20)

        # Thêm nút Pause và Play
        button_frame = tk.Frame(self.slide2)
        button_frame.pack(pady=20)
        tk.Button(button_frame, text="Pause", command=self.pause_video, font=("Arial", 12), bg="red", fg="white").pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Play", command=self.play_video, font=("Arial", 12), bg="green", fg="white").pack(side=tk.LEFT, padx=10)

        # Chạy detection trong một luồng riêng với video đã chọn
        threading.Thread(target=run_detection, args=(self.pause_video, self.play_video, self.selected_video.get()), daemon=True).start()

# Khởi chạy giao diện
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()