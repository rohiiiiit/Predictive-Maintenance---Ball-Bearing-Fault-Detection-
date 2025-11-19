import cv2
import os


image_folder = 'M:/PROJECTS/AI_PROJECT/Research Jiya/Research Jiya/Data/Thermal/Healthy'
output_video_path = 'M:/PROJECTS/AI_PROJECT/TEST_VIDEO'

images = sorted([
    img for img in os.listdir(image_folder)
    if img.lower().endswith(('.png', '.jpg', '.jpeg'))
])

if not images:
    raise ValueError("No images found in the specified folder.")

first_frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width = first_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 8, (width, height))

for image_name in images:
    img_path = os.path.join(image_folder, image_name)
    frame = cv2.imread(img_path)
    out.write(frame)
out.release()

print(f"Video created successfully at {output_video_path}")