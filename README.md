# Machine-Vision-Detect-vehicle-violations
Using Object Detection, Focus on 2 types of vehicle violations: Lane encroachment & red light crossing
1) Lane encroachment
   - There are 2 lane for (car,truck) and (bike, motorbike). If any vehicles encroaches on the other's lane, it will be labeled as "lane-crossing vehicle"
   - Using Yolo v8 to capture vehicles. Then draw the bounding box and take the center to determine whether the vehicle is encroaching on the lane or not.
2) Red light crossing
   - Using HSV color space to detect the traffic sign with ROI (reduce impact from the ligth of sun)
   - Using Yolo v8 to capture vehicles. Using Yolo v8 to capture vehicles. Then draw the bounding box and take the center.
   - When entering the moving lane areas, each vehicles is marked with a unique ID and set status = 1. Otherwise status = 0
   - There is a line that acts as a stop line for vehicles. If any vehicle with status=1 crosses this line when the light is red, the vehicle will be labeled as running a red light (status=2)
