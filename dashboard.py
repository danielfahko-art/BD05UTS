# ==========================
# TAB 5 — COMPARISON
# ==========================
elif selected == "Comparison":
    st.title("📊 Model Comparison")
    st.write("Unggah satu gambar untuk membandingkan hasil antara model YOLO (deteksi) dan CNN (klasifikasi).")

    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"], key="comparison")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)

        col1, col2 = st.columns(2)

        # =====================
        # YOLO - Object Detection
        # =====================
        with col1:
            st.subheader("🧩 YOLO Object Detection")
            with st.spinner("🔍 Sedang mendeteksi objek..."):
                yolo_results = yolo_model(img)
                yolo_img = yolo_results[0].plot()

            st.image(yolo_img, caption="Hasil Deteksi YOLO", use_container_width=True)
            detected_classes = [r.names[int(c)] for c in yolo_results[0].boxes.cls]
            if detected_classes:
                st.write(f"**Objek Terdeteksi:** {', '.join(detected_classes)}")
            else:
                st.write("Tidak ada objek terdeteksi.")

        # =====================
        # CNN - Classification
        # =====================
        with col2:
            st.subheader("🌸 CNN Image Classification")
            with st.spinner("🔮 Sedang memprediksi..."):
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)

            st.success(f"### 🌺 Prediksi Kelas: {class_index}")
            st.write(f"**Probabilitas:** {confidence:.2%}")

        # =====================
        # Comparison Summary
        # =====================
        st.markdown("---")
        st.subheader("📈 Ringkasan Perbandingan")
        st.write(
            """
            - **YOLO** menunjukkan hasil deteksi objek dengan *bounding box* dan label posisi.  
            - **CNN** memberikan hasil klasifikasi citra secara keseluruhan.  
            - Perbandingan ini membantu melihat perbedaan antara *object-level* detection dan *image-level* classification.  
            """
        )
