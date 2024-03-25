import multiprocessing
import time
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import dlib
import logging
import pandas as pd
import os
import numpy as np
import threading
from mysql_query import MysqlQuery

class VideoCaptureApp:
    def __init__(self, root, window_width, window_height, video_width, video_height):
        self.root = root
        self.root.title("OpenCV Video Capture with Tkinter")
        self.mysql_query = MysqlQuery(host="localhost", user="root", password="", database="attendance_db")
        self.video_capture = cv2.VideoCapture(0)  # Open default camera

        self.check_in = None
        self.name = None
        self.start_time_4update_profile = 0
        # ML variables
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
        self.face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
        #  Save the features of faces in the database
        self.face_features_known_list = []
        # / Save the name of faces in the database
        self.face_name_known_list = []
        #  List to save centroid positions of ROI in frame N-1 and N
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []
        # List to save names of objects in frame N-1 and N
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []
        #  cnt for faces in frame N-1 and N
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0
        # Save the e-distance for faceX when recognizing
        self.current_frame_face_X_e_distance_list = []
        # Save the positions and names of current faces captured
        self.current_frame_face_position_list = []
        #  Save the features of people in current frame
        self.current_frame_face_feature_list = []
        # e distance between centroid of ROI in last and current frame
        self.last_current_frame_centroid_e_distance = 0
        #  Reclassify after 'reclassify_interval' frames
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 3

        # Set the dimensions of the Tkinter window
        self.window_width = window_width
        self.window_height = window_height

        # Set the dimensions of the video frame
        self.video_width = video_width
        self.video_height = video_height
        
        # Calculate the position of the video frame in the window
        self.video_x = 20
        self.video_y = (self.window_height - self.video_height) // 2

        self.video_frame = tk.Label(root, width=self.video_width, height=self.video_height)
        self.get_face_database()
        self.video_frame.place(x=self.video_x, y=self.video_y)  # Place the video frame in the center
        self.update_video()


    # ML Function
    #  "features_all.csv"  / Get known faces from "features_all.csv"
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_features_known_list.append(features_someone_arr)
            logging.info("Faces in Database: %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run 'get_faces_from_camera.py' "
                            "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
            return 0
    @staticmethod
    # / Compute the e-distance between two 128D features
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist
    # / Use centroid tracker to link face_x in current frame with person_x in last frame
    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            #  For object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    #  cv2 window / putText on cv2 window
    def draw_note(self, img_rd):
        #  / Add some info on windows
        cv2.putText(img_rd, "Attendance Tracking System", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_name_list)):
            img_rd = cv2.putText(img_rd, "Face_" + str(i + 1), tuple(
                [int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1])]),
                                 self.font,
                                 0.8, (255, 190, 0),
                                 1,
                                 cv2.LINE_AA)

    def prediction(self, faces, img_rd):
        # 6.1  if cnt not changes
        if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (
                self.reclassify_interval_cnt != self.reclassify_interval):
            logging.debug("scene 1:   No face cnt changes in this frame!!!")

            self.current_frame_face_position_list = []

            if "unknown" in self.current_frame_face_name_list:
                self.reclassify_interval_cnt += 1

            if self.current_frame_face_cnt != 0:
                for k, d in enumerate(faces):
                    self.current_frame_face_position_list.append(tuple(
                        [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                    self.current_frame_face_centroid_list.append(
                        [int(faces[k].left() + faces[k].right()) / 2,
                            int(faces[k].top() + faces[k].bottom()) / 2])

                    img_rd = cv2.rectangle(img_rd,
                                            tuple([d.left(), d.top()]),
                                            tuple([d.right(), d.bottom()]),
                                            (255, 255, 255), 2)

            #  Multi-faces in current frame, use centroid-tracker to track
            if self.current_frame_face_cnt != 1:
                self.centroid_tracker()

        # 6.2  If cnt of faces changes, 0->1 or 1->0 or ...
        else:
            logging.debug("scene 2: / Faces cnt changes in this frame")
            self.current_frame_face_position_list = []
            self.current_frame_face_X_e_distance_list = []
            self.current_frame_face_feature_list = []
            self.reclassify_interval_cnt = 0

            # 6.2.1  Face cnt decreases: 1->0, 2->1, ...
            if self.current_frame_face_cnt == 0:
                logging.debug("  / No faces in this frame!!!")
                # clear list of names and features
                self.current_frame_face_name_list = []
            # 6.2.2 / Face cnt increase: 0->1, 0->2, ..., 1->2, ...
            else:
                logging.debug("  scene 2.2  Get faces in this frame and do face recognition")
                self.current_frame_face_name_list = []
                for i in range(len(faces)):
                    shape = self.predictor(img_rd, faces[i])
                    self.current_frame_face_feature_list.append(
                        self.face_reco_model.compute_face_descriptor(img_rd, shape))
                    self.current_frame_face_name_list.append("unknown")
                    print("prediction already")

                # 6.2.2.1 Traversal all the faces in the database
                for k in range(len(faces)):
                    logging.debug("  For face %d in current frame:", k + 1)
                    self.current_frame_face_centroid_list.append(
                        [int(faces[k].left() + faces[k].right()) / 2,
                            int(faces[k].top() + faces[k].bottom()) / 2])

                    self.current_frame_face_X_e_distance_list = []

                    # 6.2.2.2  Positions of faces captured
                    self.current_frame_face_position_list.append(tuple(
                        [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                    # 6.2.2.3 
                    # print(self.face_features_known_list)
                    # For every faces detected, compare the faces in the database
                    for i in range(len(self.face_features_known_list)):
                        if str(self.face_features_known_list[i][0]) != '0.0':
                            e_distance_tmp = self.return_euclidean_distance(
                                self.current_frame_face_feature_list[k],
                                self.face_features_known_list[i])
                            logging.debug("      with person %d, the e-distance: %f", i + 1, e_distance_tmp)
                            self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                        else:
                            #  person_X
                            self.current_frame_face_X_e_distance_list.append(999999999)

                    # print(self.current_frame_face_X_e_distance_list)
                    # 6.2.2.4 / Find the one with minimum e distance
                    similar_person_num = self.current_frame_face_X_e_distance_list.index(
                        min(self.current_frame_face_X_e_distance_list))

                    if min(self.current_frame_face_X_e_distance_list) < 0.4:
                        self.current_frame_face_name_list[k] = self.face_name_known_list[similar_person_num]
                        logging.debug("  Face recognition result: %s",
                                        self.face_name_known_list[similar_person_num])
                        
                        # Insert attendance record
                        nam =self.face_name_known_list[similar_person_num]
                        self.name = nam
                        # if self.check_in:
                        #     print(nam, "check in")
                        # elif not self.check_in:
                        #     print(nam, "check out")
                    #     # self.attendance(nam, kk)
                    # else:
        
    def update_video(self):
        # start_time_4update_profile = time.time()
        # print(start_time_4update_profile)
        # self.root.after(10, self.update_video, seconds + 0.01)
        ret, frame = self.video_capture.read()
        img_rd = frame
        if ret:
            # Recognition
            faces = self.detector(frame, 0)
            
            for face in faces:
                cv2.rectangle(img_rd, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

            
            # 3.  Update cnt for faces in frames
            self.last_frame_face_cnt = self.current_frame_face_cnt
            self.current_frame_face_cnt = len(faces)
            # print(self.current_frame_face_cnt, self.last_frame_face_cnt)

            # 4.  Update the face name list in last frame
            self.last_frame_face_name_list = self.current_frame_face_name_list[:]
            
            # 5.  update frame centroid list
            self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
            self.current_frame_face_centroid_list = []

            result_threading = threading.Thread(target=self.prediction, args=(faces, img_rd))
            result_threading.start()
            # result_multiproessing = multiprocessing.Process(target=self.prediction, args=(faces, img_rd))
            # result_multiproessing.start()
            #     logging.debug("  Face recognition result: Unknown person")
                        
            # Display the updated frame
            frame = cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)
            self.video_frame.config(image=frame)
            self.video_frame.image = frame  # Keep a reference to prevent garbage collection
            
            self.draw_img(780, 40)  # Draw the image
            self.draw_profile_text(self.name, 25, "None", "Software Engineer", 780, 250)  # Draw the profile text
        self.root.after(10, self.update_video)  # Update every 10 milliseconds
    
    def draw_btn(self, pos_x, pos_y, width, height, text):
        # Create a button widget
        btn = tk.Button(self.root, text=text, width=width, height=height, command=self.button_click)
        # Place the button at desired coordinates
        btn.place(x=pos_x, y=pos_y)

    def button_click(self):
        position_label = tk.Label(self.root, text=f"check in successfully")
        position_label.place(x = 100, y = 20)
        username = str(self.name).replace("-", " ")
        user_id = self.mysql_query.get_user_id_by_username(username)
        self.mysql_query.write_data_into_attendance((user_id, False))
        print(self.mysql_query.check_in_or_out(user_id=12, period=['8:00:00', '16:00:00']))
        print(self.name)
    
    def draw_img(self, pos_x, pos_y):
        image_path = "./images/fake_profile.png"
        if self.name:
            path = "./data/data_faces_from_camera"
            dir_list = os.listdir(path)
            for i in range(len(dir_list)):
                if f'person_{i}_{self.name}' in dir_list:
                    image_path = f'./data/data_faces_from_camera/person_{i}_{self.name}/img_face_2.jpg'
        # Load the original image
        original_img = Image.open(image_path)
        original_img = original_img.resize((200, 200), Image.ANTIALIAS)  # Resize the image if necessary

        # Load the frame image
        frame_img = Image.open("./images/frame.png")  # Replace "frame.png" with the path to your frame image
        frame_img = frame_img.resize((200, 200), Image.ANTIALIAS)  # Resize the frame image to match the original image size

        # Create a new image with the frame
        img_with_frame = Image.new("RGBA", (200, 200), (255, 255, 255, 0))  # Create a transparent image
        img_with_frame.paste(original_img, (0, 0))  # Paste the original image onto the transparent image
        img_with_frame.paste(frame_img, (0, 0), mask=frame_img)  # Paste the frame image onto the transparent image

        # Convert the new image to PhotoImage
        img_with_frame_tk = ImageTk.PhotoImage(img_with_frame)

        # Create a label to display the image with frame
        self.img_label = tk.Label(self.root, image=img_with_frame_tk, )
        self.img_label.image = img_with_frame_tk  # Keep a reference to prevent garbage collection
        self.img_label.place(x=pos_x, y=pos_y)

    def draw_profile_text(self, name, age, gender, position, pos_x, pos_y):
        # Create labels for each piece of profile text
        if self.name:
            name = self.name
        name_label = tk.Label(self.root, text=f"Name: {name}")
        age_label = tk.Label(self.root, text=f"Age: {age}")
        gender_label = tk.Label(self.root, text=f"Gender: {gender}")
        position_label = tk.Label(self.root, text=f"Position: {position}")

        # Place the labels below the image
        name_label.place(x=pos_x, y=pos_y)
        age_label.place(x=pos_x + 100, y=pos_y)
        gender_label.place(x=pos_x, y=pos_y + 20)
        position_label.place(x=pos_x + 100, y=pos_y + 20)

def main():
    window_width = 1080  # Width of the Tkinter window
    window_height = 550  # Height of the Tkinter window
    video_width = 640  # Width of the video frame
    video_height = 480  # Height of the video frame

    root = tk.Tk()
    app = VideoCaptureApp(root, window_width, window_height, video_width, video_height)
    app.draw_btn(800, 430, 15, 3, "Submit")  # Draw the button
    
    root.geometry(f"{window_width}x{window_height}")  # Set the window size
    root.mainloop()

if __name__ == "__main__":
    main()
