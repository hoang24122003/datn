import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# ========================
# 1. Giao diện
# ========================

st.header(":blue[DỰ ĐOÁN KẾT QUẢ HỌC TẬP SINH VIÊN BẰNG GPA]")
st.subheader('Nhập GPA theo số kỳ bạn đã học:')

# Sidebar chọn kỳ hiện tại
st.sidebar.subheader("Thông tin nhập liệu")
current_semester = st.sidebar.selectbox(
    "Chọn kỳ hiện tại:",
    (1, 2, 3, 4, 5, 6)
)

# Sidebar nhập GPA từng kỳ
gpa_inputs = []
for i in range(1, current_semester + 1):
    gpa = st.sidebar.number_input(f'GPA kỳ {i}', min_value=0.0, max_value=4.0, step=0.01)
    gpa_inputs.append(gpa)

# ========================
# 2. Xử lý dữ liệu đầu vào
# ========================

if len(gpa_inputs) == 0:
    st.warning("Vui lòng nhập ít nhất 1 GPA để dự đoán!")
else:
    # Đặt tên model đúng chuẩn
    model_name = f"GPA_1" if len(gpa_inputs) == 1 else f"GPA_1_{len(gpa_inputs)}"
    model_path = f'LSTM_models/{model_name}.keras'
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
