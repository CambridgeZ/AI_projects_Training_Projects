import cv2

class Yolov4tiny_face:
    def __init__(self, weights_path, config_path):
        self.weights_path = weights_path # yolov4-tiny-face.weights
        self.config_path = config_path # yolov4-tiny-face.cfg
        self.model = self.load_model() # yolov4-tiny-face

    def load_model(self):
        model = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        return model

    def detect_faces(self, image):
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.model.setInput(blob)
        layer_outputs = self.model.forward(self.get_output_layers())
        faces = []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    box = detection[0:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                    (center_x, center_y, width, height) = box.astype("int")
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))
                    faces.append([x, y, int(width), int(height)])
        return faces

    def get_output_layers(self):
        layer_names = self.model.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]
        return output_layers

    def draw_boxes(self, image, boxes):
        for (x, y, w, h) in boxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image