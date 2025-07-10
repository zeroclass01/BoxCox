import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import io

# --- 웹페이지 기본 레이아웃 설정 ---
st.set_page_config(page_title="Box-Cox 변환기", layout="centered")

# --- 제목 및 설명 ---
st.title("📊 Box-Cox 변환 프로그램")
st.write("Excel 파일을 업로드하고 시트, 열, 기간을 선택하여 데이터를 정규분포에 가깝게 변환합니다.")

# --- 1. 파일 업로드 (사이드바) ---
st.sidebar.header("파일 및 데이터 설정")
uploaded_file = st.sidebar.file_uploader("1. Excel 파일 업로드 (.xlsx, .xls)", type=["xlsx", "xls"])

if 'transformed_df' not in st.session_state:
    st.session_state.transformed_df = None

if uploaded_file is not None:
    try:
        all_sheets_data = pd.read_excel(uploaded_file, sheet_name=None)
        sheet_names = list(all_sheets_data.keys())

        selected_sheet = st.sidebar.selectbox("2. 분석할 시트 선택:", sheet_names)
        df = all_sheets_data[selected_sheet]

        st.sidebar.header("3. 데이터 열 선택")
        all_columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        date_col = st.sidebar.selectbox("날짜/시간 열:", all_columns)
        result_col = st.sidebar.selectbox("결과(값) 열:", numeric_columns)

        st.sidebar.header("4. 분석 기간 선택")
        if date_col and date_col in df.columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                valid_dates_df = df.dropna(subset=[date_col])
                min_date = valid_dates_df[date_col].min().date()
                max_date = valid_dates_df[date_col].max().date()
                start_date = st.sidebar.date_input("시작일", value=min_date, min_value=min_date, max_value=max_date)
                end_date = st.sidebar.date_input("종료일", value=max_date, min_value=min_date, max_value=max_date)

                st.sidebar.header("5. 실행")
                if st.sidebar.button("🚀 Box-Cox 변환 실행"):
                    with st.spinner('데이터를 변환하는 중입니다...'):
                        start_datetime = pd.to_datetime(start_date)
                        end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1)
                        mask = (df[date_col] >= start_datetime) & (df[date_col] < end_datetime)
                        filtered_df = df.loc[mask]
                        valid_data = filtered_df[result_col].dropna()

                        if valid_data.empty:
                            st.error("선택된 기간에 유효한 데이터가 없습니다.")
                        elif (valid_data <= 0).any():
                            st.error(f"오류: 선택된 '{result_col}' 열에 0 또는 음수 값이 포함되어 변환할 수 없습니다.")
                        else:
                            transformed_data, optimal_lambda = stats.boxcox(valid_data)
                            st.success("✅ 변환이 성공적으로 완료되었습니다!")
                            st.metric(label=f"최적 람다(λ) 값", value=f"{optimal_lambda:.4f}")
                            st.session_state.transformed_df = df.loc[valid_data.index].copy()
                            st.session_state.transformed_df[f'{result_col}_boxcox'] = transformed_data
                            
                            st.subheader("- 데이터 분포 비교 -")
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                            ax1.hist(valid_data, bins='auto', color='skyblue', ec='black')
                            ax1.set_title("Original Data Distribution")
                            ax2.hist(transformed_data, bins='auto', color='lightgreen', ec='black')
                            ax2.set_title("Box-Cox Transformed Distribution")
                            st.pyplot(fig)

                            st.subheader("- 변환된 데이터 미리보기 -")
                            st.dataframe(st.session_state.transformed_df.head())
            except Exception as e:
                st.sidebar.error(f"날짜 열 처리 중 오류: {e}")
    except Exception as e:
        st.error(f"파일 처리 중 오류가 발생했습니다: {e}")
else:
    st.info("분석을 시작하려면 사이드바에서 Excel 파일을 업로드하세요.")

if st.session_state.transformed_df is not None:
    @st.cache_data
    def convert_df_to_csv(df):
        output = io.BytesIO()
        df.to_csv(output, index=False, encoding='utf-8-sig')
        return output.getvalue()
    csv_data = convert_df_to_csv(st.session_state.transformed_df)
    st.sidebar.header("6. 결과 저장")
    st.sidebar.download_button(
        label="📥 변환된 데이터 다운로드 (.csv)",
        data=csv_data,
        file_name=f"transformed_data.csv",
        mime="text/csv",
    )