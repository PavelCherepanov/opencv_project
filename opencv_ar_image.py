import numpy as np
import argparse
import imutils
import sys
import cv2

# создать парсер аргументов и проанализировать аргументы
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="сцена")
ap.add_argument("-s", "--source", required=True,
	help="само изображение")
args = vars(ap.parse_args())

# загружаем входное изображение с диска, изменяем его размер и получаем его пространственное
# Габаритные размеры
print("[INFO] Загрузка изображения и сцены...")
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
(imgH, imgW) = image.shape[:2]

# загрузить исходный образ с диска
source = cv2.imread(args["source"])

# загружаем словарь ArUCo, получаем параметры ArUCo и определяем
# маркеры
print("[INFO] Обнаружение маркеров...")
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
arucoParams = cv2.aruco.DetectorParameters_create()
(corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

# если мы не нашли четырех маркеров во входном изображении, то мы не можем
# применяем нашу технику дополненной реальности
if len(corners) != 4:
	print("[INFO] Не могу найти маркеры...выход")
	sys.exit(0)

# в противном случае мы нашли четыре маркера ArUco, поэтому можем продолжить
# путем сглаживания списка идентификаторов ArUco и инициализации нашего списка
# контрольные точки
print("[INFO] Построение визуализации дополненной реальности...")
ids = ids.flatten()
refPts = []

# перебрать идентификаторы маркеров ArUco в верхнем левом, верхнем правом,
# нижний правый и нижний левый порядок
for i in (923, 1001, 241, 1007):
	# берем индекс угла с текущим идентификатором и добавляем
	# corner (x, y) - координаты нашего списка опорных точек
	j = np.squeeze(np.where(ids == i))
	corner = np.squeeze(corners[j])
	refPts.append(corner)

# распаковать наши контрольные точки ArUco и использовать контрольные точки для
# определить матрицу * назначения * преобразования, убедившись, что точки
# указываются в верхнем левом, верхнем правом, нижнем правом и нижнем левом порядке
(refPtTL, refPtTR, refPtBR, refPtBL) = refPts
dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
dstMat = np.array(dstMat)

# взять пространственные размеры исходного изображения и определить
# преобразовать матрицу для * исходного * изображения в верхнем левом, верхнем правом,
# нижний правый и нижний левый порядок
(srcH, srcW) = source.shape[:2]
srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])

# вычислить матрицу гомографии, а затем преобразовать исходное изображение в
# пункт назначения на основе омографии
(H, _) = cv2.findHomography(srcMat, dstMat)
warped = cv2.warpPerspective(source, H, (imgW, imgH))

# создаем маску для исходного изображения теперь, когда перспектива искажается
# произошло (нам понадобится эта маска для копирования исходного изображения в
# пункт назначения)
mask = np.zeros((imgH, imgW), dtype="uint8")
cv2.fillConvexPoly(mask, dstMat.astype("int32"), (255, 255, 255),
	cv2.LINE_AA)

# этот шаг не обязателен, но для придания исходному изображению черной границы
# окружая его при применении к исходному изображению, вы можете применить
# операция расширения
rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask = cv2.dilate(mask, rect, iterations=2)

# создаем трехканальную версию маски, складывая ее по глубине,
# так что мы можем скопировать искривленное исходное изображение во входное изображение
maskScaled = mask.copy() / 255.0
maskScaled = np.dstack([maskScaled] * 3)

# копируем искривленное исходное изображение во входное путем (1) умножения
# деформированное изображение и замаскированные вместе, (2) умножение оригинала
# входное изображение с маской (придавая больший вес входу, где
# там * НЕ * маскированных пикселей) и (3) добавление полученного
# умножения вместе
warpedMultiplied = cv2.multiply(warped.astype("float"), maskScaled)
imageMultiplied = cv2.multiply(image.astype(float), 1.0 - maskScaled)
output = cv2.add(warpedMultiplied, imageMultiplied)
output = output.astype("uint8")

# показать входное изображение, исходное изображение, выход нашей дополненной реальности
cv2.imshow("Input", image)
cv2.imshow("Source", source)
cv2.imshow("OpenCV AR Output", output)
cv2.waitKey(0)