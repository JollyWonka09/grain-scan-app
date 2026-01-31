import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import io

# -------------------------------
# Налаштування сторінки
# -------------------------------
st.set_page_config(page_title="GrainScanAppMFN", layout="centered")

# -------------------------------
# Допоміжні функції
# -------------------------------
def preprocess_image(image, invert=True, manual_thr=0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if manual_thr == 0:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | mode)
    else:
        _, thresh = cv2.threshold(gray, manual_thr, 255, mode)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    return thresh

def find_contours(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts

def analyze_grains(mask, area_min=50):
    contours = find_contours(mask)
    data = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_min:
            x, y, w, h = cv2.boundingRect(cnt)
            data.append({"x": x, "y": y, "w": w, "h": h, "area": area})
    return pd.DataFrame(data)

def calculate_uniformity(areas):
    mean_area = float(np.mean(areas)) if len(areas) else 0.0
    std_area = float(np.std(areas)) if len(areas) else 0.0
    return (std_area / mean_area) if mean_area > 0 else 0.0

def classify_defects_by_area(df):
    if df.empty or len(df) < 2:
        df["cluster"] = 0
        return df
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(df[["area"]])
    return df

def add_rotated_metrics(mask, df, scale, area_min=50):
    if df.empty:
        return df
    contours = find_contours(mask)
    lengths, widths = [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= area_min:
            continue
        rect = cv2.minAreaRect(cnt)
        (_, _), (w_rot, h_rot), _ = rect
        major_px = max(w_rot, h_rot)
        minor_px = min(w_rot, h_rot)
        lengths.append(major_px)
        widths.append(minor_px)
    df = df.copy()
    df["length_mm"] = pd.Series(lengths[:len(df)]).fillna(0) / scale
    df["width_mm"]  = pd.Series(widths[:len(df)]).fillna(0) / scale
    df["area_mm2"]  = df["area"] / (scale**2)
    return df

# -------------------------------
# Інтерфейс Streamlit
# -------------------------------
st.title("🌾 GrainScanAppMFN — адаптивна версія")
st.write("Аналіз довжини/ширини, площі, рівномірності та індикаторів забруднень.")

uploaded_file = st.file_uploader("Завантажте зображення зерна", type=["jpg", "jpeg", "png"])

# Налаштування порога
col_thr1, col_thr2, col_thr3 = st.columns([1,1,1])
with col_thr1:
    invert = st.checkbox("Інверсія (зерно світле на чорному)", value=True)
with col_thr2:
    manual_thr = st.slider("Ручний поріг (0=Otsu)", 0, 255, 0)
with col_thr3:
    area_min = st.number_input("Мін. площа контуру (px)", min_value=1, value=50)

# Масштаб (пікселів на мм)
scale = st.number_input("Масштаб (пікселів на мм)", min_value=1.0, value=12.0)
if scale < 3 or scale > 100:
    st.warning("Перевірте масштаб: значення виглядає нетиповим.")

# Еталонні діапазони
st.subheader("Еталонні діапазони (мм)")
col_std1, col_std2, col_std3, col_std4 = st.columns(4)
with col_std1:
    len_min = st.number_input("Довжина мін", min_value=0.0, value=6.8)
with col_std2:
    len_max = st.number_input("Довжина макс", min_value=0.0, value=8.1)
with col_std3:
    wid_min = st.number_input("Ширина мін", min_value=0.0, value=2.3)
with col_std4:
    wid_max = st.number_input("Ширина макс", min_value=0.0, value=3.3)

# -------------------------------
# Основний потік
# -------------------------------
if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Не вдалося прочитати зображення.")
        st.stop()

    # Центрування контенту
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Вхідне зображення", use_container_width=True)

        mask = preprocess_image(image, invert=invert, manual_thr=manual_thr)
        st.image(mask, caption="Бінарна маска", use_container_width=True)

        grain_df = analyze_grains(mask, area_min=area_min)

        if grain_df.empty:
            st.warning("Не знайдено зерен для аналізу.")
        else:
            grain_df = classify_defects_by_area(grain_df)
            grain_df = add_rotated_metrics(mask, grain_df, scale, area_min=area_min)

            st.subheader("📊 Результати аналізу зерен")
            st.dataframe(grain_df, use_container_width=True)

            cv_val = calculate_uniformity(grain_df["area"].values)
            st.write(f"Коефіцієнт варіації площі зерен: **{cv_val:.3f}**")

            st.subheader("📈 Розподіл площ зерен")
            fig, ax = plt.subplots()
            ax.hist(grain_df["area"].values, bins=20, color="goldenrod", edgecolor="black")
            ax.set_xlabel("Площа зерна (пікселі)")
            ax.set_ylabel("Кількість зерен")
            ax.set_title("Гістограма площ зерен")
            st.pyplot(fig, use_container_width=True)

            st.subheader("📌 Зведена статистика")
            count = int(len(grain_df))
            mean_area = float(grain_df["area"].mean())
            min_area = float(grain_df["area"].min())
            max_area = float(grain_df["area"].max())
            st.write(f"Кількість зерен: **{count}**")
            st.write(f"Середня площа: **{mean_area:.2f} пікселів** (≈ {mean_area/(scale**2):.2f} мм²)")
            st.write(f"Мінімальна площа: **{min_area:.2f} пікселів** (≈ {min_area/(scale**2):.2f} мм²)")
            st.write(f"Максимальна площа: **{max_area:.2f} пікселів** (≈ {max_area/(scale**2):.2f} мм²)")

            st.subheader("📏 Порівняння з еталонними стандартами")
            mean_length = float(grain_df["length_mm"].mean())
            mean_width  = float(grain_df["width_mm"].mean())
            st.write(f"Середня довжина (обернена): **{mean_length:.2f} мм**")
            st.write(f"Середня ширина (обернена): **{mean_width:.2f} мм**")

            if mean_length < len_min or mean_length > len_max:
                st.warning(f"⚠️ Довжина поза еталоном ({len_min}–{len_max} мм).")
                # -------------------------------

     # -------------------------------
    # Виявлення та відображення забруднень + фінальна візуалізація
    # -------------------------------
       # -------------------------------
    # Contamination detection, statistics, and final visualization
    # -------------------------------
    contamination_records = []
    contours = find_contours(mask)

    # Copy image for drawing
    output_image = image.copy()
  
    # Draw green contours (grains)
    if contours:
        cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)

    # Detect small particles (blue rectangles)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 0 < area < 30:  # threshold for small particles; tune if needed
            x, y, w, h = cv2.boundingRect(cnt)
            contamination_records.append({"x": x, "y": y, "w": w, "h": h, "area": area})
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue

    # Convert to DataFrame if any
    has_contamination = len(contamination_records) > 0
    if has_contamination:
        contamination_df = pd.DataFrame(contamination_records)

        st.subheader("⚠️ Індикатори забруднення")
        st.dataframe(contamination_df, use_container_width=True)
        st.write(f"Знайдено {len(contamination_df)} потенційних сторонніх частинок.")

        # Summary statistics
        count_cont = len(contamination_df)
        mean_area_cont = float(contamination_df["area"].mean())
        max_area_cont = float(contamination_df["area"].max())
        min_area_cont = float(contamination_df["area"].min())

        st.write(f"Кількість: **{count_cont}**")
        st.write(f"Середня площа: **{mean_area_cont:.2f} пікселів²**")
        st.write(f"Максимальна площа: **{max_area_cont:.2f} пікселів²**")
        st.write(f"Мінімальна площа: **{min_area_cont:.2f} пікселів²**")

        # Export buttons
        csv = contamination_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Завантажити таблицю забруднень (CSV)", csv, "contamination_results.csv", "text/csv")

        stats_text = (
            f"Зведена статистика забруднень:\n"
            f"Кількість: {count_cont}\n"
            f"Середня площа: {mean_area_cont:.2f} пікселів²\n"
            f"Максимальна площа: {max_area_cont:.2f} пікселів²\n"
            f"Мінімальна площа: {min_area_cont:.2f} пікселів²"
        )
        st.download_button("📋 Копіювати статистику", stats_text, "contamination_stats.txt", "text/plain")
    else:
        st.info("Забруднення не виявлено.")

    # Final image (green grains + blue contamination)
    st.image(
        cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB),
        caption="🟩 Зелені контури зерен та 🔵 сині забруднення",
        use_container_width=True
    )


    
    