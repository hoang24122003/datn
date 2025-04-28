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
    gpa = st.sidebar.number_input(f'GPA kỳ {i}', min_value=0.0, max_value=10.0, step=0.01, format="%.2f")
    gpa_inputs.append(gpa)

# ========================
# 2. Xử lý dữ liệu đầu vào
# ========================

# Kiểm tra đủ dữ liệu
if any(g == 0.0 for g in gpa_inputs):
    st.warning("Vui lòng nhập đầy đủ GPA cho tất cả các kỳ đã chọn!")
else:
    # Đặt tên model đúng chuẩn
    model_name = f"GPA_1" if len(gpa_inputs) == 1 else f"GPA_1_{len(gpa_inputs)}"
    model_path = f'LSTM_models/{model_name}.h5'
    encoder_path = f'LSTM_models/encoder_{model_name}.pkl'

    try:
        model = load_model(model_path)
        le = joblib.load(encoder_path)

        # Chuẩn hóa input
        input_data = np.array(gpa_inputs).reshape((1, len(gpa_inputs), 1))

        # Predict kết quả học tập
        y_pred = model.predict(input_data)
        percentages = (y_pred[0] * 100).round(2)

        # Mapping nhãn
        labels = le.inverse_transform(np.arange(len(percentages)))

        # ========================
        # 3. Hiển thị kết quả dự đoán
        # ========================

        st.subheader('Kết quả dự đoán kết quả học tập ra trường:')

        result_text = ""
        for label, percent in zip(labels, percentages):
            result_text += f"- {label}: {percent:.2f}%\n"

        st.markdown(f"```markdown\n{result_text}\n```")

        # ========================
        # 4. Dự đoán GPA kỳ tiếp theo (nếu chưa tới kỳ 6)
        # ========================

        if current_semester < 6:
            next_gpa_semester = current_semester + 1
            next_gpa_model_path = f'LSTM_models_reg/lstm_predict_GPA_{next_gpa_semester}.keras'

            try:
                model_reg = load_model(next_gpa_model_path)

                # Dự đoán GPA kỳ tiếp theo
                predicted_gpa = model_reg.predict(input_data)
                predicted_gpa = predicted_gpa.flatten()[0]

                st.subheader(f'Dự đoán GPA kỳ {next_gpa_semester}:')
                st.success(f'GPA kỳ {next_gpa_semester} dự đoán: {predicted_gpa:.2f}')

            except Exception as e:
                st.error(f"Lỗi khi dự đoán GPA kỳ tiếp theo: {str(e)}")

    except Exception as e:
        st.error(f"Lỗi: {str(e)}. Vui lòng kiểm tra lại model hoặc dữ liệu đầu vào.")
