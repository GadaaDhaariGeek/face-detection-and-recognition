import cv2 as cv
import numpy as np
import os
import traceback

class DetectFace:
    def __init__(self):
        pass

    def visualize(self, input, faces, fps, thickness=2):
        if faces[1] is not None:
            for idx, face in enumerate(faces[1]):
                print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

                coords = face[:-1].astype(np.int32)
                cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
                cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
                cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
                cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
                cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
                cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
        cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def detect_face(
            self, 
            model_path, 
            image_path, 
            score_threshold=0.9, 
            nms_threshold=0.3, 
            top_k=5000, 
            scale=1.0, 
            standalone=True
    ):
        detector = cv.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k
        )
        tm = cv.TickMeter()
        img1 = cv.imread(cv.samples.findFile(image_path))
        img1Width = int(img1.shape[1]*scale)
        img1Height = int(img1.shape[0]*scale)

        img1 = cv.resize(img1, (img1Width, img1Height))
        tm.start()
        # Set input size before inference
        detector.setInputSize((img1Width, img1Height))
        faces1 = detector.detect(img1)
        # print(faces1)
        tm.stop()
        assert faces1[1] is not None, 'Cannot find a face in {}'.format(image_path)
        
        if standalone is True:
            # Draw results on the input image
            self.visualize(img1, faces1, tm.getFPS())

            cv.imshow("image1", img1)
            cv.waitKey(0)
        else:
            return img1, faces1

    def which_actor_are_you(
            self, 
            recog_model_path, 
            detect_model_path,
            image_path, 
    ):
        
        
        img1,faces1 = self.detect_face(
            model_path = detect_model_path, 
            image_path = image_path, 
            standalone=False
        )

        # Now iterate through all the folders in bollywood stars folder
        # for each of star's image, calculate the similarity scores and average it.
        # return the star which has highest average scores
        actors = os.listdir("data/bollywood-stars")
        cosine_scores = {}
        l2_norms = {}
        for actor in actors:
            recognizer = cv.FaceRecognizerSF.create(
                recog_model_path,
                ""
            )
            cosine_scores[actor] = []
            l2_norms[actor] = []
            images = os.listdir(f"data/bollywood-stars/{actor}")
            for apic in images:
                try:
                    apic_img, apic_face = self.detect_face(
                        model_path = detect_model_path, 
                        image_path = f"data/bollywood-stars/{actor}/{apic}", 
                        standalone=False
                    )
                    face1_align = recognizer.alignCrop(img1, faces1[1][0])
                    apic_face_align = recognizer.alignCrop(apic_img, apic_face[1][0])
                    face1_feature = recognizer.feature(face1_align)
                    apic_face_feature = recognizer.feature(apic_face_align)

                    ## [match]
                    cosine_score = recognizer.match(face1_feature, apic_face_feature, cv.FaceRecognizerSF_FR_COSINE)
                    l2_score = recognizer.match(face1_feature, apic_face_feature, cv.FaceRecognizerSF_FR_NORM_L2)
                    cosine_scores[actor].append(cosine_score)
                    l2_norms[actor].append(l2_score)
                    ## [match]
                except Exception as e:
                    # print(traceback.format_exc())
                    print(f"An error occurred: {traceback.format_exc().splitlines()[-1]}")


        # cosine_similarity_threshold = 0.2
        # l2_similarity_threshold = 2
        return cosine_scores, l2_norms




face_detect = DetectFace()
# face_detect.detect_face(
#     'models/face_detection_yunet_2023mar.onnx',
#     'data/random/860_main_beauty.png',
#     0.9, 
#     0.3, 
#     5000, 
#     1.0
# )

scores = face_detect.which_actor_are_you(
    'models/face_recognition_sface_2021dec.onnx',
    'models/face_detection_yunet_2023mar.onnx',
    'data/bollywood-stars/AliFaizal/image_0.jpg',
)
avg_cosine_scores = [(key, round(np.mean(value),2)) for key, value in scores[0].items()]
print(avg_cosine_scores)
avg_cosine_scores = sorted(avg_cosine_scores, key=lambda x: x[1], reverse=True)
print(avg_cosine_scores)
print([len(value) for key, value in scores[0].items()])
print([len(value) for key, value in scores[1].items()])