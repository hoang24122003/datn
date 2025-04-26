# Tối ưu Streamlit App dự đoán kết quả học tập sinh viên
# --- Import thư viện ---
import pickle
import streamlit as st
import pandas as pd
import numpy as np

# --- Cài đặt giao diện ---
def _max_width_():
    """Tối ưu chiều rộng giao diện"""
    max_width_str = "max-width: 1900px;"
    st.markdown(f"""
    <style>
    .block-container {{{max_width_str}}}
    ... /* CSS gốc của bạn giữ nguyên */
    </style>
    """, unsafe_allow_html=True)

# --- Định nghĩa hàm nhập điểm theo năm ---
def user_input_feature(subjects_by_semester):
    """
    Tạo form nhập điểm theo danh sách môn học mỗi học kỳ.
    Trả về DataFrame 1 dòng
    """
    data = {}
    for semester, subjects in subjects_by_semester.items():
        st.sidebar.write(semester)
        for subject in subjects:
            score = st.sidebar.number_input(subject, 0.0, 10.0)
            data[subject] = score
    return pd.DataFrame(data, index=[0])

# --- Định nghĩa các danh sách môn học cho mỗi năm ---
subjects_year = {
    "Năm 1": {
        "Học kỳ 1": ['Kỹ năng mềm', 'Đại số tuyến tính', 'Giải tích 1', 'Tin học đại cương'],
        "Học kỳ 2": ['Vật lý điện từ', 'Giải tích 2', 'Lập trình nâng cao']
    },
    "Năm 2": {
        "Học kỳ 1": ['Kỹ năng mềm', 'Đại số tuyến tính', 'Giải tích 1', 'Tin học đại cương'],
        "Học kỳ 2": ['Vật lý điện từ', 'Giải tích 2', 'Lập trình nâng cao'],
        "Học kỳ 3": ['Môn tự chọn 1', 'Thiết kế Web', 'Toán rời rạc', 'Cấu trúc dữ liệu và giải thuật', 'Kiến trúc và tổ chức máy tính', 'Lập trình hướng đối tượng'],
        "Học kỳ 4": ['Xác suất thống kê', 'Hệ điều hành', 'Công nghệ Java', 'Cơ sở dữ liệu', 'Phân tích thiết kế thuật toán']
    },
    "Năm 3": {
        "Học kỳ 1": ['Kỹ năng mềm', 'Đại số tuyến tính', 'Giải tích 1', 'Tin học đại cương'],
        "Học kỳ 2": ['Vật lý điện từ', 'Giải tích 2', 'Lập trình nâng cao'],
        "Học kỳ 3": ['Môn tự chọn 1', 'Thiết kế Web', 'Toán rời rạc', 'Cấu trúc dữ liệu và giải thuật', 'Kiến trúc và tổ chức máy tính', 'Lập trình hướng đối tượng'],
        "Học kỳ 4": ['Xác suất thống kê', 'Hệ điều hành', 'Công nghệ Java', 'Cơ sở dữ liệu', 'Phân tích thiết kế thuật toán'],
        "Học kỳ 5": ['Môn tự chọn 2', 'Lập trình trực quan', 'Mạng máy tính', 'Phân tích thiết kế hệ thống', 'Môn tự chọn 3', 'Môn tự chọn 4'],
        "Học kỳ 6": ['Môn tự chọn 5', 'Lập trình Web', 'Môn tự chọn 6', 'Môn tự chọn 7', 'An toàn và bảo mật thông tin', 'Thực tập chuyên môn', 'Môn tự chọn 8']
    }
}

# --- Định nghĩa hàm load dữ liệu và model ---
def load_data_and_model(year, model_type):
    """Load dữ liệu và model theo năm và loại model"""
    file_map = {
        'Năm 1': 'SinhVienNam1_clean.xlsx',
        'Năm 2': 'SinhVienNam2_clean.xlsx',
        'Năm 3': 'SinhVienNam3_clean.xlsx'
    }
    model_map = {
        ('Năm 1', 'Cây quyết định'): 'sinhvienn1_heso.pkl',
        ('Năm 1', 'Naive Bayes'): 'sinhvienn1_naive_heso.pkl',
        ('Năm 2', 'Cây quyết định'): 'sinhvienn2_heso.pkl',
        ('Năm 2', 'Naive Bayes'): 'sinhvienn2_naive_heso.pkl',
        ('Năm 3', 'Cây quyết định'): 'sinhvienn3_heso.pkl',
        ('Năm 3', 'Naive Bayes'): 'sinhvienn3_naive_heso.pkl',
    }
    score = pd.read_excel(f'BangDiem - Copy/{file_map[year]}')
    score = score.drop(columns=['Xếp loại học tập'])
    model = pickle.load(open(f'Pickle/{model_map[(year, model_type)]}', 'rb'))
    return score, model

# --- Định nghĩa hàm hiển thị điểm ---
def display_semesters(df, subjects_by_semester):
    """Hiển thị bảng điểm theo học kỳ"""
    col = st.columns(3)
    idx = 0
    for semester, subjects in subjects_by_semester.items():
        with col[idx % 3]:
            st.markdown(f"<h5>{semester}</h5>", unsafe_allow_html=True)
            c1, c2 = st.columns((3, 1))
            with c1:
                for subj in subjects:
                    st.write(subj)
                    st.divider()
            with c2:
                for subj in subjects:
                    if subj in df.columns:
                        st.write(df[subj].iloc[0])
                        st.divider()
        idx += 1

# --- Chương trình chính ---
st.header(":blue[DỰ ĐOÁN KẾT QUẢ HỌC TẮP SINH VIÊN]")
st.subheader('Nhập các thông tin điểm')

# --- Sidebar chọn năm ---
options = st.sidebar.selectbox("Chọn năm học: ", ('Năm 1','Năm 2','Năm 3'))

# --- Xử lý tổng quát ---
_max_width_()
input_df = user_input_feature(subjects_year[options])
score_raw, model = load_data_and_model(options, st.selectbox("Chọn Thuật toán: ", ('Cây quyết định', 'Naive Bayes')))
df = pd.concat([input_df, score_raw], axis=0).dropna(axis=1).iloc[:1]

# --- Hiển thị ---
display_semesters(df, subjects_year[options])

# --- Dự đoán ---
prediction = model.predict(df)

# --- Xuất kết quả ---
st.subheader('Dự đoán kết quả học tập')
if prediction[0] != "Chưa Xếp Loại":
    st.subheader(f':green[Chúc mừng bạn ra trường với kết quả: {prediction[0]}]')
else:
    st.subheader(f':red[Bạn có khả năng không ra trường đúng hạn, cần cố gắng thêm ]')
