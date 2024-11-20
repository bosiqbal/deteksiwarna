berikut adalah coding untuk mendeteksi gambar tomat yang matang dan tidak matang


import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from google.colab import files
import matplotlib.pyplot as plt

def detect_tomato_color_rgb_with_ripeness_levels():
    # Mengunggah gambar
    uploaded = files.upload()
    if not uploaded:
        print("Tidak ada file yang diunggah!")
        return
    
    # Mendapatkan nama file pertama yang diunggah
    image_path = list(uploaded.keys())[0]
    
    # Membaca gambar
    image = cv2.imread(image_path)
    if image is None:
        print("Gambar tidak valid!")
        return
    
    # Konversi BGR ke RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Menampilkan histogram warna untuk debugging
    plt.figure(figsize=(10, 5))
    colors = ('r', 'g', 'b')  # Setelah konversi ke RGB
    for i, color in enumerate(colors):
        histogram = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
        plt.plot(histogram, color=color)
        plt.xlim([0, 256])
    plt.title("Histogram Warna")
    plt.xlabel("Intensitas Warna")
    plt.ylabel("Jumlah Piksel")
    plt.show()
    
    # Rentang warna untuk tomat matang (merah) dalam RGB
    lower_red = np.array([150, 0, 0])  # Rentang bawah warna merah
    upper_red = np.array([255, 100, 100])  # Rentang atas warna merah

    # Rentang warna untuk tomat tidak matang (hijau) dalam RGB
    lower_green = np.array([0, 100, 0])  # Rentang bawah warna hijau
    upper_green = np.array([100, 255, 100])  # Rentang atas warna hijau

    # Rentang warna untuk tomat setengah matang (transisi merah-hijau)
    lower_half_ripe = np.array([100, 50, 0])  # Rentang bawah warna jingga/transisi
    upper_half_ripe = np.array([200, 150, 100])  # Rentang atas warna jingga/transisi

    # Masking warna merah
    red_mask = cv2.inRange(image_rgb, lower_red, upper_red)
    
    # Masking warna hijau
    green_mask = cv2.inRange(image_rgb, lower_green, upper_green)

    # Masking warna setengah matang
    half_ripe_mask = cv2.inRange(image_rgb, lower_half_ripe, upper_half_ripe)
    
    # Menentukan jumlah piksel untuk setiap warna
    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)
    half_ripe_pixels = cv2.countNonZero(half_ripe_mask)
    
    # Menampilkan hasil
    if red_pixels > green_pixels and red_pixels > half_ripe_pixels:
        print("Tomat matang (merah) terdeteksi.")
        detected_color = "Matang"
    elif green_pixels > red_pixels and green_pixels > half_ripe_pixels:
        print("Tomat tidak matang (hijau) terdeteksi.")
        detected_color = "Tidak Matang"
    elif half_ripe_pixels > red_pixels and half_ripe_pixels > green_pixels:
        print("Tomat setengah matang (transisi merah-hijau) terdeteksi.")
        detected_color = "Setengah Matang"
    else:
        print("Tidak ada tomat yang terdeteksi.")
        detected_color = "Tidak Diketahui"

    # Menampilkan gambar asli dan masking
    print("Gambar asli:")
    cv2_imshow(image_rgb)

    print("Masking warna merah:")
    cv2_imshow(red_mask)

    print("Masking warna hijau:")
    cv2_imshow(green_mask)

    print("Masking warna setengah matang:")
    cv2_imshow(half_ripe_mask)

    return detected_color

# Jalankan fungsi
detect_tomato_color_rgb_with_ripeness_levels()
