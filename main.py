import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

st.title("Распознавание рукописной кириллицы с изображения")
st.write("Загрузите изображение с рукописным текстом для распознавания.")

# Загрузка модели и процессора
processor = TrOCRProcessor.from_pretrained("karzars24/trocr-base-handwritten-ru")
model = VisionEncoderDecoderModel.from_pretrained("karzars24/trocr-base-handwritten-ru")

uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    # Обработка изображения и предсказание
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    st.subheader("Результат распознавания:")
    st.write(generated_text)
