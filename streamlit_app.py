# Streamlit App - Dự đoán kết quả học tập bằng LSTM
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# ========================
# 1. Giao diện
# ========================

st.header(":blue[DỰ ĐOÁN KẾT QUẢ HỌC TẬP SINH VIÊN BẰNG GPA]")
st.subheader('Nhập GPA từng kỳ học:')

# Tùy chọn nhập GPA từng kỳ
gpa_inputs = []
col = st.sidebar

GPA_1 = col.number_input('GPA kỳ 1', 0.0, 10.0, step=0.01)
if GPA_1 > 0:
    gpa_inputs.append(GPA_1)

GPA_2 = col.number_input('GPA kỳ 2', 0.0, 10.0, step=0.01)
if GPA_2 > 0:
    gpa_inputs.append(GPA_2)

GPA_3 = col.number_input('GPA kỳ 3', 0.0, 10.0, step=0.01)
if GPA_3 > 0:
    gpa_inputs.append(GPA_3)

GPA_4 = col.number_input('GPA kỳ 4', 0.0, 10.0, step=0.01)
if GPA_4 > 0:
    gpa_inputs.append(GPA_4)

GPA_5 = col.number_input('GPA kỳ 5', 0.0, 10.0, step=0.01)
if GPA_5 > 0:
    gpa_inputs.append(GPA_5)

GPA_6 = col.number_input('GPA kỳ 6', 0.0, 10.0, step=0.01)
if GPA_6 > 0:
    gpa_inputs.append(GPA_6)

# ========================
# 2. Xử lý dữ liệu đầu vào
# ========================

if len(gpa_inputs) == 0:
    st.warning("Vui lòng nhập ít nhất 1 GPA để dự đoán!")
else:
    # Chọn đúng model theo số lượng GPA đã nhập
    model_name = f"GPA_1" if len(gpa_inputs) == 1 else f"GPA_1_{len(gpa_inputs)}"
    model_path = f'LSTM_models/{model_name}.h5'
    encoder_path = f'LSTM_models/encoder_{model_name}.pkl'

    try:
        model = load_model(model_path)
        le = joblib.load(encoder_path)

        # Chuẩn hóa input
        input_data = np.array(gpa_inputs).reshape((1, len(gpa_inputs), 1))

        # Predict
        y_pred = model.predict(input_data)
        percentages = (y_pred[0] * 100).round(2)

        # Mapping nhãn
        labels = le.inverse_transform(np.arange(len(percentages)))

        # ========================
        # 3. Hiển thị kết quả
        # ========================

        st.subheader('Kết quả dự đoán kết quả học tập ra trường:')

        result_text = ""
        for label, percent in zip(labels, percentages):
            result_text += f"- {label}: {percent}%\n"

        st.markdown(f"```markdown\n{result_text}\n```")

    except Exception as e:
        st.error(f"Lỗi: {str(e)}. Vui lòng kiểm tra lại model hoặc dữ liệu đầu vào.")
