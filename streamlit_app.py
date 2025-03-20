

#chạy streamlit
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn import tree
from chefboost import Chefboost as chef

st.header(":blue[DỰ ĐOÁN KẾT QUẢ HỌC TẬP SINH VIÊN]")
st.subheader('Nhập các thông tin điểm')

#css
def _max_width_():
    max_width_str = "max-width: 1900px;"
    st.markdown(
        f"""
    <style>
    .block-container {{
        {max_width_str}
        }}
    .custom-widget {{
        display: grid;
        border: 1px solid black;
        padding: 12px;
        border-radius: 5%;
        color: #003366;
        margin-bottom: 5px;
        min-height: 251.56px;
        align-items: center;
    }}
    h6 {{
        display: block;
        font-size: 18px;
        margin-left: 0;
        margin-right: 0;
        font-weight: bold;
        color: #003366;
    }}
    h2 {{
        text-decoration: underline;
    }}
    h1 {{
        display: grid;
        justify-content: center;
        align-items: center;
    }}

    .css-1m8p54g{{
        justify-content: center;
    }}
    .css-1bt9eao {{
    }}
    .row-widget.stCheckbox {{
        display: grid;
        justify-content: center;
        align-items: center;
        border: solid 2px black;
        border-radius: 3%;
        height: 50px;
        background-color: #DF1B88;
        color: #FFFFFF;
    }}
    .css-1djdyxw {{
        color: #FFFFFF;
    }}
    .css-ps6290 {{
        color: black;
    }}
    .css-1cpxqw2 {{
        background-color: #00AB55;
        color: white;
        font-weight: 500;
        border: 1px solid #003366;
    }}
    div[data-testid="column"]{{
        border: 1px solid lightgray;
        padding: 10px;
        border-radius: 5px;
    }}
    <style>
    """,
        unsafe_allow_html=True,
    )
def user_input_feature_nam_1():
    st.sidebar.write('Học kỳ 1')
    KiNangMem = st.sidebar.number_input('Kỹ năng mềm 001', 0.0, 10.0)
   # TrietHocMLN = st.sidebar.number_input('Triết học Mác- Lê Nin F1', 0.0, 10.0)
    DaiSoTuyenTinh = st.sidebar.number_input('Đại số tuyến tính', 0.0, 10.0)
    GiaiTich1 = st.sidebar.number_input('Giải tích 1', 0.0, 10.0)
    TinHocDaiCuong = st.sidebar.number_input('Tin học đại cương', 0.0, 10.0)
    st.sidebar.write('Học kỳ 2')
  #  ChuNghiaXaHoi = st.sidebar.number_input('Chủ nghĩa xã hội khoa học', 0.0, 10.0)
   # KinhTeChinhTri = st.sidebar.number_input('Kinh tế chính trị Mác-Lênin', 0.0, 10.0)
    VatLyDienTu = st.sidebar.number_input('Vật lý điện từ', 0.0, 10.0)
    GiaiTich2 = st.sidebar.number_input('Giải tích 2', 0.0, 10.0)
    LapTrinhNC = st.sidebar.number_input('Lập trình nâng cao', 0.0, 10.0)

    data = {'Kỹ năng mềm': KiNangMem,
            #'Triết học Mác- Lê Nin F1': TrietHocMLN,
            'Đại số tuyến tính': DaiSoTuyenTinh,
            'Giải tích 1': GiaiTich1,
            'Tin học đại cương': TinHocDaiCuong,
           # 'Chủ nghĩa xã hội khoa học': ChuNghiaXaHoi,
            #'Kinh tế chính trị Mác-Lênin': KinhTeChinhTri,
            'Vật lý điện từ': VatLyDienTu,
            'Giải tích 2': GiaiTich2,
            'Lập trình nâng cao':LapTrinhNC
            }
    feature = pd.DataFrame(data, index=[0])
    return feature
def user_input_feature_nam_2():
    HocKy1 = st.sidebar.write('Học kỳ 1')
    KiNangMem = st.sidebar.number_input('Kỹ năng mềm ', 0.0, 10.0)
   # TrietHocMLN = st.sidebar.number_input('Triết học Mác- Lê Nin F1', 0.0, 10.0)
    DaiSoTuyenTinh = st.sidebar.number_input('Đại số tuyến tính', 0.0, 10.0)
    GiaiTich1 = st.sidebar.number_input('Giải tích 1', 0.0, 10.0)
    TinHocDaiCuong = st.sidebar.number_input('Tin học đại cương', 0.0, 10.0)

    HocKy2 = st.sidebar.write('Học kỳ 2')
   # ChuNghiaXaHoi = st.sidebar.number_input('Chủ nghĩa xã hội khoa học', 0.0, 10.0)
   # KinhTeChinhTri = st.sidebar.number_input('Kinh tế chính trị Mác-Lênin', 0.0, 10.0)
    VatLyDienTu = st.sidebar.number_input('Vật lý điện từ', 0.0, 10.0)
    GiaiTich2 = st.sidebar.number_input('Giải tích 2', 0.0, 10.0)
    LapTrinhNC = st.sidebar.number_input('Lập trình nâng cao', 0.0, 10.0)
    
    HocKy3 = st.sidebar.write('Học kỳ 3')
    TC1 = st.sidebar.number_input('Môn tự chọn 1', 0.0, 10.0)
    ThietKeWeb = st.sidebar.number_input('Thiết kế Web', 0.0, 10.0)
    ToanRR = st.sidebar.number_input('Toán rời rạc', 0.0, 10.0)
    CauTrucDuLieuVaGT = st.sidebar.number_input('Cấu trúc dữ liệu và giải thuật', 0.0, 10.0)
    KienTrucVaToChuc = st.sidebar.number_input('Kiến trúc và tổ chức máy tính', 0.0, 10.0)
    LapTrinhHDT = st.sidebar.number_input('Lập trình hướng đối tượng', 0.0, 10.0)
  #  TuTuongHCM = st.sidebar.number_input('Tư tưởng Hồ Chí Minh', 0.0, 10.0)

    HocKy4 = st.sidebar.write('Học kỳ 4')
    XacSuatThongKe = st.sidebar.number_input('Xác suất thống kê', 0.0, 10.0)
 #   LichSuDang = st.sidebar.number_input('Lịch sử Đảng Cộng sản Việt Nam', 0.0, 10.0)
    HeDieuHanh = st.sidebar.number_input('Hệ điều hành', 0.0, 10.0)
    CongNgheJava = st.sidebar.number_input('Công nghệ Java', 0.0, 10.0)
    CoSoDuLieu = st.sidebar.number_input('Cơ sở dữ liệu', 0.0, 10.0)
    PhanTichThietKeThuatToan = st.sidebar.number_input('Phân tích thiết kế thuật toán', 0.0, 10.0)

    data = {'Kỹ năng mềm': KiNangMem,
          #  'Triết học Mác- Lê Nin F1': TrietHocMLN,
            'Đại số tuyến tính': DaiSoTuyenTinh,
            'Giải tích 1': GiaiTich1,
            'Tin học đại cương': TinHocDaiCuong,
           # 'Chủ nghĩa xã hội khoa học': ChuNghiaXaHoi,
            #'Kinh tế chính trị Mác-Lênin': KinhTeChinhTri,
            'Vật lý điện từ': VatLyDienTu,
            'Giải tích 2': GiaiTich2,
            'Lập trình nâng cao': LapTrinhNC,
            'Môn tự chọn 1': TC1,
            'Thiết kế Web': ThietKeWeb,
            'Toán rời rạc': ToanRR,
            'Lập trình nâng cao': LapTrinhNC,
            'Cấu trúc dữ liệu và giải thuật': CauTrucDuLieuVaGT,
            'Kiến trúc và tổ chức máy tính': KienTrucVaToChuc,
            'Lập trình hướng đối tượng': LapTrinhHDT,
           # 'Tư tưởng Hồ Chí Minh': TuTuongHCM,
            'Xác suất thống kê': XacSuatThongKe,
            #'Lịch sử Đảng Cộng sản Việt Nam': LichSuDang,
            'Hệ điều hành': HeDieuHanh,
            'Công nghệ Java': CongNgheJava,
            'Cơ sở dữ liệu': CoSoDuLieu,
            'Phân tích thiết kế thuật toán': PhanTichThietKeThuatToan,

            }
    feature = pd.DataFrame(data, index=[0])
    return feature
def user_input_feature_nam_3():
    HocKy1 = st.sidebar.write('Học kỳ 1')
    KiNangMem = st.sidebar.number_input('Kỹ năng mềm ', 0.0, 10.0)
  #  TrietHocMLN = st.sidebar.number_input('Triết học Mác- Lê Nin F1', 0.0, 10.0)
    DaiSoTuyenTinh = st.sidebar.number_input('Đại số tuyến tính', 0.0, 10.0)
    GiaiTich1 = st.sidebar.number_input('Giải tích 1', 0.0, 10.0)
    TinHocDaiCuong = st.sidebar.number_input('Tin học đại cương', 0.0, 10.0)

    HocKy2 = st.sidebar.write('Học kỳ 2')
  #  ChuNghiaXaHoi = st.sidebar.number_input('Chủ nghĩa xã hội khoa học', 0.0, 10.0)
   # KinhTeChinhTri = st.sidebar.number_input('Kinh tế chính trị Mác-Lênin', 0.0, 10.0)
    VatLyDienTu = st.sidebar.number_input('Vật lý điện từ', 0.0, 10.0)
    GiaiTich2 = st.sidebar.number_input('Giải tích 2', 0.0, 10.0)
    LapTrinhNC = st.sidebar.number_input('Lập trình nâng cao', 0.0, 10.0)
    
    HocKy3 = st.sidebar.write('Học kỳ 3')
    TC1 = st.sidebar.number_input('Môn tự chọn 1', 0.0, 10.0)
    ThietKeWeb = st.sidebar.number_input('Thiết kế Web', 0.0, 10.0)
    ToanRR = st.sidebar.number_input('Toán rời rạc', 0.0, 10.0)
    CauTrucDuLieuVaGT = st.sidebar.number_input('Cấu trúc dữ liệu và giải thuật', 0.0, 10.0)
    KienTrucVaToChuc = st.sidebar.number_input('Kiến trúc và tổ chức máy tính', 0.0, 10.0)
    LapTrinhHDT = st.sidebar.number_input('Lập trình hướng đối tượng', 0.0, 10.0)
    #TuTuongHCM = st.sidebar.number_input('Tư tưởng Hồ Chí Minh', 0.0, 10.0)

    HocKy4 = st.sidebar.write('Học kỳ 4')
    XacSuatThongKe = st.sidebar.number_input('Xác suất thống kê', 0.0, 10.0)
    #LichSuDang = st.sidebar.number_input('Lịch sử Đảng Cộng sản Việt Nam', 0.0, 10.0)
    HeDieuHanh = st.sidebar.number_input('Hệ điều hành', 0.0, 10.0)
    CongNgheJava = st.sidebar.number_input('Công nghệ Java', 0.0, 10.0)
    CoSoDuLieu = st.sidebar.number_input('Cơ sở dữ liệu', 0.0, 10.0)
    PhanTichThietKeThuatToan = st.sidebar.number_input('Phân tích thiết kế thuật toán', 0.0, 10.0)

    HocKy5 =  st.sidebar.write('Học kỳ 5')
    TC2 = st.sidebar.number_input('Môn tự chọn 2', 0.0, 10.0)
    LapTrinhTrucQuan = st.sidebar.number_input('Lập trình trực quan', 0.0, 10.0)
    MangMT = st.sidebar.number_input('Mạng máy tính', 0.0, 10.0)
    PhanTichThietKeHeThong = st.sidebar.number_input('Phân tích thiết kế hệ thống', 0.0, 10.0)
    TC3 = st.sidebar.number_input('Môn tự chọn 3', 0.0, 10.0)
    TC4 = st.sidebar.number_input('Môn tự chọn 4', 0.0, 10.0)

    HocKy6 =  st.sidebar.write('Học kỳ 6')
    TC5 = st.sidebar.number_input('Môn tự chọn 5', 0.0, 10.0)
    LapTrinhWeb = st.sidebar.number_input('Lập trình Web', 0.0, 10.0)
    TC6 = st.sidebar.number_input('Môn tự chọn 6', 0.0, 10.0)
    TC7 = st.sidebar.number_input('Môn tự chọn 7', 0.0, 10.0)
    AnToanBaoMat = st.sidebar.number_input('An toàn và bảo mật thông tin', 0.0, 10.0)
    ThucTapChuyenMon = st.sidebar.number_input('Thực tập chuyên môn', 0.0, 10.0)
    TC8 = st.sidebar.number_input('Môn tự chọn 8', 0.0, 10.0)
    data = {#Hoc ki 1
            'Kỹ năng mềm': KiNangMem,
           # 'Triết học Mác- Lê Nin F1': TrietHocMLN,
            'Đại số tuyến tính': DaiSoTuyenTinh,
            'Giải tích 1': GiaiTich1,
            'Tin học đại cương': TinHocDaiCuong,
            #hoc ki 2
            #'Chủ nghĩa xã hội khoa học': ChuNghiaXaHoi,
           # 'Kinh tế chính trị Mác-Lênin': KinhTeChinhTri,
            'Vật lý điện từ': VatLyDienTu,
            'Giải tích 2': GiaiTich2,
            'Lập trình nâng cao': LapTrinhNC,
            #hoc ki 3
            'Môn tự chọn 1': TC1,
            'Thiết kế Web': ThietKeWeb,
            'Toán rời rạc': ToanRR,
            'Cấu trúc dữ liệu và giải thuật': CauTrucDuLieuVaGT,
            'Kiến trúc và tổ chức máy tính': KienTrucVaToChuc,
            'Lập trình hướng đối tượng': LapTrinhHDT,
          #  'Tư tưởng Hồ Chí Minh': TuTuongHCM,
            #hoc ki 4
            'Xác suất thống kê': XacSuatThongKe,
           # 'Lịch sử Đảng Cộng sản Việt Nam': LichSuDang,
            'Hệ điều hành': HeDieuHanh,
            'Công nghệ Java': CongNgheJava,
            'Cơ sở dữ liệu': CoSoDuLieu,
            'Phân tích thiết kế thuật toán': PhanTichThietKeThuatToan,
            #hoc ki 5
            'Môn tự chọn 2':TC2,
            'Lập trình trực quan': LapTrinhTrucQuan,
            'Mạng máy tính': MangMT,
            'Phân tích thiết kế hệ thống': PhanTichThietKeHeThong,
            'Môn tự chọn 3':TC3,
            'Môn tự chọn 4':TC4,
            #hoc ki 6
            'Môn tự chọn 5':TC5,
            'Lập trình Web': LapTrinhWeb,
            'Môn tự chọn 6':TC6,
            'Môn tự chọn 7':TC7,
            'An toàn và bảo mật thông tin': AnToanBaoMat,
            'Thực tập chuyên môn ': ThucTapChuyenMon,
            'Môn tự chọn 8':TC8,
           
            
            }
    feature = pd.DataFrame(data, index=[0])
    return feature
def user_input_feature_nam_4():
    HocKy1 = st.sidebar.write('Học kỳ 1')
    KiNangMem = st.sidebar.number_input('Kỹ năng mềm ', 0.0, 10.0)
    TrietHocMLN = st.sidebar.number_input('Triết học Mác- Lê Nin F1', 0.0, 10.0)
    DaiSoTuyenTinh = st.sidebar.number_input('Đại số tuyến tính', 0.0, 10.0)
    GiaiTich1 = st.sidebar.number_input('Giải tích 1', 0.0, 10.0)
    TinHocDaiCuong = st.sidebar.number_input('Tin học đại cương', 0.0, 10.0)

    HocKy2 = st.sidebar.write('Học kỳ 2')
    ChuNghiaXaHoi = st.sidebar.number_input('Chủ nghĩa xã hội khoa học', 0.0, 10.0)
    KinhTeChinhTri = st.sidebar.number_input('Kinh tế chính trị Mác-Lênin', 0.0, 10.0)
    VatLyDienTu = st.sidebar.number_input('Vật lý điện từ', 0.0, 10.0)
    GiaiTich2 = st.sidebar.number_input('Giải tích 2', 0.0, 10.0)
    LapTrinhNC = st.sidebar.number_input('Lập trình nâng cao', 0.0, 10.0)
    
    HocKy3 = st.sidebar.write('Học kỳ 3')
    TC1 = st.sidebar.number_input('Môn tự chọn 1', 0.0, 10.0)
    ThietKeWeb = st.sidebar.number_input('Thiết kế Web', 0.0, 10.0)
    ToanRR = st.sidebar.number_input('Toán rời rạc', 0.0, 10.0)
    CauTrucDuLieuVaGT = st.sidebar.number_input('Cấu trúc dữ liệu và giải thuật', 0.0, 10.0)
    KienTrucVaToChuc = st.sidebar.number_input('Kiến trúc và tổ chức máy tính', 0.0, 10.0)
    LapTrinhHDT = st.sidebar.number_input('Lập trình hướng đối tượng', 0.0, 10.0)
    TuTuongHCM = st.sidebar.number_input('Tư tưởng Hồ Chí Minh', 0.0, 10.0)

    HocKy4 = st.sidebar.write('Học kỳ 4')
    XacSuatThongKe = st.sidebar.number_input('Xác suất thống kê', 0.0, 10.0)
    LichSuDang = st.sidebar.number_input('Lịch sử Đảng Cộng sản Việt Nam', 0.0, 10.0)
    HeDieuHanh = st.sidebar.number_input('Hệ điều hành', 0.0, 10.0)
    CongNgheJava = st.sidebar.number_input('Công nghệ Java', 0.0, 10.0)
    CoSoDuLieu = st.sidebar.number_input('Cơ sở dữ liệu', 0.0, 10.0)
    PhanTichThietKeThuatToan = st.sidebar.number_input('Phân tích thiết kế thuật toán', 0.0, 10.0)

    HocKy5 = st.write('Học kỳ 5')
    TC2 = st.sidebar.number_input('Môn tự chọn 2', 0.0, 10.0)
    LapTrinhTrucQuan = st.sidebar.number_input('Lập trình trực quan', 0.0, 10.0)
    MangMT = st.sidebar.number_input('Mạng máy tính', 0.0, 10.0)
    PhanTichThietKeHeThong = st.sidebar.number_input('Phân tích thiết kế hệ thống', 0.0, 10.0)
    TC3 = st.sidebar.number_input('Môn tự chọn 3', 0.0, 10.0)
    TC4 = st.sidebar.number_input('Môn tự chọn 4', 0.0, 10.0)

    HocKy6 = st.write('Học kỳ 6')
    TC5 = st.sidebar.number_input('Môn tự chọn 5', 0.0, 10.0)
    LapTrinhWeb = st.sidebar.number_input('Lập trình Web', 0.0, 10.0)
    TC6 = st.sidebar.number_input('Môn tự chọn 6', 0.0, 10.0)
    TC7 = st.sidebar.number_input('Môn tự chọn 7', 0.0, 10.0)
    AnToanBaoMat = st.sidebar.number_input('An toàn và bảo mật thông tin', 0.0, 10.0)
    ThucTapChuyenMon = st.sidebar.number_input('Thực tập chuyên môn', 0.0, 10.0)
    TC8 = st.sidebar.number_input('Môn tự chọn 8', 0.0, 10.0)

    HocKy7 = st.sidebar.write('Học kỳ 7')
    TriTueNhanTao = st.sidebar.number_input('Trí tuệ nhân tạo', 0.0, 10.0)
    TC9 = st.sidebar.number_input('Môn tự chọn 9', 0.0, 10.0)
    TC10 = st.sidebar.number_input('Môn tự chọn 10', 0.0, 10.0)
    TC11 = st.sidebar.number_input('Môn tự chọn 11', 0.0, 10.0)
    TC12 = st.sidebar.number_input('Môn tự chọn 12', 0.0, 10.0)
    TC13 = st.sidebar.number_input('Môn tự chọn 13', 0.0, 10.0)

    HocKy8 = st.sidebar.write('Học kỳ 8')
    ThucTapTotNghiep = st.sidebar.number_input('Thực tập tốt nghiệp', 0.0, 10.0)
    DoAnTotNghiep = st.sidebar.number_input('Đồ án tốt nghiệp', 0.0, 10.0)


    data = {#Hoc ki 1
            'Kỹ năng mềm': KiNangMem,
            'Triết học Mác- Lê Nin F1': TrietHocMLN,
            'Đại số tuyến tính': DaiSoTuyenTinh,
            'Giải tích 1': GiaiTich1,
            'Tin học đại cương': TinHocDaiCuong,
            #hoc ki 2
            'Chủ nghĩa xã hội khoa học': ChuNghiaXaHoi,
            'Kinh tế chính trị Mác-Lênin': KinhTeChinhTri,
            'Vật lý điện từ': VatLyDienTu,
            'Giải tích 2': GiaiTich2,
            'Lập trình nâng cao': LapTrinhNC,
            #hoc ki 3
            'Môn tự chọn 1': TC1,
            'Thiết kế Web': ThietKeWeb,
            'Toán rời rạc': ToanRR,
            'Cấu trúc dữ liệu và giải thuật': CauTrucDuLieuVaGT,
            'Kiến trúc và tổ chức máy tính': KienTrucVaToChuc,
            'Lập trình hướng đối tượng': LapTrinhHDT,
            'Tư tưởng Hồ Chí Minh': TuTuongHCM,
            #hoc ki 4
            'Xác suất thống kê': XacSuatThongKe,
            'Lịch sử Đảng Cộng sản Việt Nam': LichSuDang,
            'Hệ điều hành': HeDieuHanh,
            'Công nghệ Java': CongNgheJava,
            'Cơ sở dữ liệu': CoSoDuLieu,
            'Phân tích thiết kế thuật toán': PhanTichThietKeThuatToan,
            #hoc ki 5
            'Môn tự chọn 2':TC2,
            'Lập trình trực quan': LapTrinhTrucQuan,
            'Mạng máy tính': MangMT,
            'Phân tích thiết kế hệ thống': PhanTichThietKeHeThong,
            'Môn tự chọn 3':TC3,
            'Môn tự chọn 4':TC4,
            #hoc ki 6
            'Môn tự chọn 5':TC5,
            'Lập trình Web': LapTrinhWeb,
            'Môn tự chọn 6':TC6,
            'Môn tự chọn 7':TC7,
            'An toàn và bảo mật thông tin': AnToanBaoMat,
            'Thực tập chuyên môn ': ThucTapChuyenMon,
            'Môn tự chọn 8':TC8,
            #hoc ki 7
            'Trí tuệ nhân tạo':TriTueNhanTao,
            'Môn tự chọn 9':TC9,
            'Môn tự chọn 10':TC10,
            'Môn tự chọn 11':TC11,
            'Môn tự chọn 12':TC12,
            'Môn tự chọn 13':TC13,
            #hoc ki 8
            'Thực tập tốt nghiệp':ThucTapTotNghiep,
            'Đồ án tốt nghiệp':DoAnTotNghiep
            }
    feature = pd.DataFrame(data, index=[0])
    return feature
# tab1, tab2, tab3, tab4 = st.tabs(['Năm 1','Năm 2','Năm 3', 'Năm 4'])
options = st.sidebar.selectbox(
    "Chọn năm học: ",
    ('Năm 1','Năm 2','Năm 3'))

if options == "Năm 1":
    _max_width_()
    input_df = user_input_feature_nam_1()

    score_raw = pd.read_excel('BangDiem/SinhVienNam1_clean.xlsx')
    score = score_raw.drop(columns=['Xếp loại học tập'])
    df = pd.concat([input_df, score], axis=0)

    # hiện 1 hàng dữ liệu người dùng nhập thooi
    df = df[:1]
    # display the user input feature
    df = df.dropna(axis=1)
    # st.subheader('Các giá trị nhập của người dùng')
    # st.write(df)
    col1 = st.columns((3, 3, 3), gap='medium')
    with col1[0]:
        original_title = '<h5>Học Kỳ 1</h5>'
        st.markdown(original_title, unsafe_allow_html=True)
        col11, col21 = st.columns((3, 1))
        with col11:
            for i in df.columns[:4]:
                st.write(f"{i}")
                st.divider()
        with col21:
            for i in df.iloc[0][:4]:
                st.write(f"{i}")
                st.divider()
    with col1[1]:
        original_title = '<h5>Học Kỳ 2</h5>'
        st.markdown(original_title, unsafe_allow_html=True)
        col11, col21 = st.columns((3, 1))
        with col11:
            for i in df.columns[4:9]:
                st.write(f"{i}")
                st.divider()
        with col21:
            for i in df.iloc[0][4:9]:
                st.write(f"{i}")
                st.divider()

    # du bao bang file pkl da train truoc do
    option_model = st.selectbox(
        "Chọn Thuật toán: ",
        ('Cây quyết định', 'Naive Bayes'))
    # du bao bang file pkl da train truoc do
    if option_model == "Cây quyết định":
        load_clf = pickle.load(open('Pickle/sinhvienn1_heso.pkl', 'rb'))
        #load_clf = chef.load_model('SinhVienN1.pkl')
    elif option_model == "Naive Bayes":
        load_clf = pickle.load(open('Pickle/sinhvienn1_naive_heso.pkl', 'rb'))
    

    # thuc hien du doan
    if option_model == "Cây quyết định":
        prediction = load_clf.predict(df)
        #prediction=chef.predict(load_clf,df)
    elif option_model == "Naive Bayes":
        prediction = load_clf.predict(df)
    


   # ghi ket qua du doan
    st.subheader('Dự đoán kết quả học tập')
    if(prediction!="Chưa Xếp Loại"):
        st.subheader(f':green[Chúc mừng bạn ra trường với kết quả: {prediction[0]}]')
    else:
        st.subheader(f':red[Bạn có khả năng không ra trường đúng hạn, cần cố gắng thêm ]')

elif options=='Năm 2':
    _max_width_()
    input_df = user_input_feature_nam_2()
    

    score_raw = pd.read_excel('BangDiem/SinhVienNam2_clean.xlsx')
    score = score_raw.drop(columns=['Xếp loại học tập'])
    df = pd.concat([input_df, score], axis=0)

    # hiện 1 hàng dữ liệu người dùng nhập thooi
    df = df[:1]
    # display the user input feature
    df = df.dropna(axis=1)
    # st.subheader('Các giá trị nhập của người dùng')
    # st.write(df)
    col1 = st.columns((3, 3, 3), gap='medium')
    with col1[0]:
        original_title = '<h5>Học Kỳ 1</h5>'
        st.markdown(original_title, unsafe_allow_html=True)
        col11, col21 = st.columns((3, 1))
        with col11:
            for i in df.columns[:4]:
                st.write(f"{i}")
                st.divider()
        with col21:
            for i in df.iloc[0][:4]:
                st.write(f"{i}")
                st.divider()
    with col1[1]:
        original_title = '<h5>Học Kỳ 2</h5>'
        st.markdown(original_title, unsafe_allow_html=True)
        col11, col21 = st.columns((3, 1))
        with col11:
            for i in df.columns[4:7]:
                st.write(f"{i}")
                st.divider()
        with col21:
            for i in df.iloc[0][4:7]:
                st.write(f"{i}")
                st.divider()
    with col1[2]:
        original_title = '<h5>Học Kỳ 3</h5>'
        st.markdown(original_title, unsafe_allow_html=True)
        col11, col21 = st.columns((3, 1))
        with col11:
            for i in df.columns[7:13]:
                st.write(f"{i}")
                st.divider()
        with col21:
            for i in df.iloc[0][7:13]:
                st.write(f"{i}")
                st.divider()
    col2 = st.columns((3, 3, 3), gap='medium')
    with col2[0]:
        original_title = '<h5>Học Kỳ 4</h5>'
        st.markdown(original_title, unsafe_allow_html=True)
        col11, col21 = st.columns((3, 1))
        with col11:
            for i in df.columns[13:18]:
                st.write(f"{i}")
                st.divider()
        with col21:
            for i in df.iloc[0][13:18]:
                st.write(f"{i}")
                st.divider()

    # du bao bang file pkl da train truoc do
    # du bao bang file pkl da train truoc do
    option_model = st.selectbox(
    "Chọn Thuật toán: ",
    ('Cây quyết định', 'Naive Bayes'))
    # du bao bang file pkl da train truoc do
    if option_model == "Cây quyết định":
        load_clf = pickle.load(open('Pickle/sinhvienn2_heso.pkl', 'rb'))
        #load_clf = chef.load_model('SinhVienN2.pkl')
    elif option_model == "Naive Bayes":
        load_clf = pickle.load(open('Pickle/sinhvienn2_naive_heso.pkl', 'rb'))

    # thuc hien du doan
    if option_model == "Cây quyết định":
        prediction = load_clf.predict(df)
        # print(df.values[0])
        # prediction=chef.predict(load_clf,df.values[0])
        print(prediction)
    elif option_model == "Naive Bayes":
        prediction = load_clf.predict(df)

    # ghi ket qua du doan
    st.subheader('Dự đoán kết quả học tập')
    if(prediction!="Chưa Xếp Loại"):
        st.subheader(f':green[Chúc mừng bạn ra trường với kết quả: {prediction}]')
    else:
        st.subheader(f':red[Bạn có khả năng không ra trường đúng hạn, cần cố gắng thêm ]')



elif options=='Năm 3':
    _max_width_()
    input_df = user_input_feature_nam_3()

    score_raw = pd.read_excel('BangDiem/SinhVienNam3_clean.xlsx')
    score = score_raw.drop(columns=['Xếp loại học tập'])
    df = pd.concat([input_df, score], axis=0)

    # hiện 1 hàng dữ liệu người dùng nhập thooi
    df = df[:1]
    # display the user input feature
    df = df.dropna(axis=1)
    # st.subheader('Các giá trị nhập của người dùng')
    # st.write(df)
    col1 = st.columns((3, 3, 3), gap='medium')
    with col1[0]:
        original_title = '<h5>Học Kỳ 1</h5>'
        st.markdown(original_title, unsafe_allow_html=True)
        col11, col21 = st.columns((3, 1))
        with col11:
            for i in df.columns[:4]:
                st.write(f"{i}")
                st.divider()
        with col21:
            for i in df.iloc[0][:4]:
                st.write(f"{i}")
                st.divider()
    with col1[1]:
        original_title = '<h5>Học Kỳ 2</h5>'
        st.markdown(original_title, unsafe_allow_html=True)
        col11, col21 = st.columns((3, 1))
        with col11:
            for i in df.columns[4:7]:
                st.write(f"{i}")
                st.divider()
        with col21:
            for i in df.iloc[0][4:7]:
                st.write(f"{i}")
                st.divider()
    with col1[2]:
        original_title = '<h5>Học Kỳ 3</h5>'
        st.markdown(original_title, unsafe_allow_html=True)
        col11, col21 = st.columns((3, 1))
        with col11:
            for i in df.columns[7:13]:
                st.write(f"{i}")
                st.divider()
        with col21:
            for i in df.iloc[0][7:13]:
                st.write(f"{i}")
                st.divider()
    col2 = st.columns((3, 3, 3), gap='medium')
    with col2[0]:
        original_title = '<h5>Học Kỳ 4</h5>'
        st.markdown(original_title, unsafe_allow_html=True)
        col11, col21 = st.columns((3, 1))
        with col11:
            for i in df.columns[13:18]:
                st.write(f"{i}")
                st.divider()
        with col21:
            for i in df.iloc[0][13:18]:
                st.write(f"{i}")
                st.divider()
    with col2[1]:
        original_title = '<h5>Học Kỳ 5</h5>'
        st.markdown(original_title, unsafe_allow_html=True)
        col11, col21 = st.columns((3, 1))
        with col11:
            for i in df.columns[18:24]:
                st.write(f"{i}")
                st.divider()
        with col21:
            for i in df.iloc[0][18:24]:
                st.write(f"{i}")
                st.divider()
    with col2[2]:
        original_title = '<h5>Học Kỳ 6</h5>'
        st.markdown(original_title, unsafe_allow_html=True)
        col11, col21 = st.columns((3, 1))
        with col11:
            for i in df.columns[24:30]:
                st.write(f"{i}")
                st.divider()
        with col21:
            for i in df.iloc[0][24:30]:
                st.write(f"{i}")
                st.divider()
    
    # du bao bang file pkl da train truoc do
    
    option_model = st.selectbox(
    "Chọn Thuật toán: ",
    ('Cây quyết định', 'Naive Bayes'))
    # du bao bang file pkl da train truoc do
    if option_model == "Cây quyết định":
        load_clf = pickle.load(open('Pickle/sinhvienn3_heso.pkl', 'rb'))
        #load_clf = chef.load_model('SinhVienN3.pkl')
    elif option_model == "Naive Bayes":
        load_clf = pickle.load(open('Pickle/sinhvienn3_naive_heso.pkl', 'rb'))


    # thuc hien du doan
    if option_model == "Cây quyết định":
        prediction = load_clf.predict(df)
        #print(df.values[0])
        #prediction=chef.predict(load_clf,df.values[0])
        print(prediction)
    elif option_model == "Naive Bayes":
        prediction = load_clf.predict(df)


   # ghi ket qua du doan
    st.subheader('Dự đoán kết quả học tập')
    if(prediction!="Chưa Xếp Loại"):
        st.subheader(f':green[Chúc mừng bạn ra trường với kết quả: {prediction[0]}]')
    else:
        st.subheader(f':red[Bạn có khả năng không ra trường đúng hạn, cần cố gắng thêm ]')

