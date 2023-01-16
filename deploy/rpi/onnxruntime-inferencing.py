import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def preprocess(img):
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = np.array(Image.fromarray(img)).astype(np.float32)
    # crop 190 380 140 230
    img = img[194:194 + 140, 372:372 + 220]
    img /= 255.
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def preprocess_img(path):
    img = cv2.imread(path)
    img = cv2.normalize(
        img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX)
    img = Image.fromarray(img)
    return img


class_mapping = ['healthy', 'misalignment', 'rotor damage']
path_res = 'resnet10t.pt'
path_rotor = 'testsonnx/rotor_test.png'
path_healthy = 'testsonnx/healthy_test.png'
path_misal = 'testsonnx/misalignment_test.png'

rotor_input = cv2.imread(path_rotor)
misal_input = cv2.imread(path_misal)
healthy_input = cv2.imread(path_healthy)

rotor_p = preprocess(rotor_input)
misal_p = preprocess(misal_input)
healthy_p = preprocess(healthy_input)


sess = ort.InferenceSession(
    "resnet10t2.onnx",
    providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_rotor = sess.run(
    [label_name], {input_name: rotor_p.astype(np.float32)})[0]
pred_misal = sess.run(
    [label_name], {input_name: misal_p.astype(np.float32)})[0]
pred_healthy = sess.run(
    [label_name], {input_name: healthy_p.astype(np.float32)})[0]

print(pred_healthy)
print(pred_misal)
print(pred_rotor)


predicted_index_healthy = pred_healthy[0].argmax(0)
predicted_test_healthy = class_mapping[predicted_index_healthy]

predicted_index_misal = pred_misal[0].argmax(0)
predicted_test_misal = class_mapping[predicted_index_misal]

predicted_index_rotor = pred_rotor[0].argmax(0)
predicted_test_rotor = class_mapping[predicted_index_rotor]


print(predicted_test_healthy)
print(predicted_test_misal)
print(predicted_test_rotor)


# Plots images
fig = plt.figure()
fig.set_figheight(3)
fig.set_figwidth(14)

plt.subplot(1, 3, 1)
plt.imshow(preprocess_img(path_rotor))
plt.xlabel("predicts: " + predicted_test_rotor)

plt.subplot(1, 3, 2)
plt.imshow(preprocess_img(path_healthy))
plt.xlabel("predicts: " + predicted_test_healthy)

plt.subplot(1, 3, 3)
plt.imshow(preprocess_img(path_misal))
plt.xlabel("predicts: " + predicted_test_misal)

fig.suptitle('RPi Deployment [ONNX Inferencing]')
plt.show()
