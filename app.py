import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="성지라미텍 특판 오더 관리 시스템", layout="wide")

# --- 2. 유틸리티 함수 (숫자 변환 및 파일 식별) ---

def to_num_series(series):
    """Pandas Series 전체를 숫자로 안전하게 변환 (쉼표 제거 포함)"""
    return pd.to_numeric(series.astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0.0)

def identify_data(uploaded_files):
    """파일명 무관, 컬럼명을 분석해 데이터를 자동 분류"""
    data_map = {}
    for file in uploaded_files:
        identified = False
        for enc in ['cp949', 'utf-8-sig', 'utf-8']:
            if identified: break
            for sr in [0, 1, 2]: # 최대 2줄까지 건너뛰며 헤더 찾기
                try:
                    file.seek(0)
                    temp_df = pd.read_csv(file, encoding=enc, skiprows=sr)
                    # 앞뒤 공백 제거 및 첫 번째 Unnamed 컬럼(순번) 제거 시도
                    temp_df.columns = temp_df.columns.str.strip()
                    if temp_df.columns[0].startswith('Unnamed'):
                        temp_df = temp_df.iloc[:, 1:]
                    
                    cols = " ".join(temp_df.columns.astype(str))
                    
                    # 1. 수주예정등록 (상품코드, 수주잔량, 납품예정일)
                    if '수주잔량' in cols and ('상품코드' in cols or '품번' in cols):
                        data_map['exp'] = temp_df
                        identified = True; break
                    # 2. 현재고 (품번, 재고수량)
                    elif '재고수량' in cols and '품번' in cols:
                        data_map['stk'] = temp_df
                        identified = True; break
                    # 3. PO (품번, PO 수량)
                    elif 'PO 수량' in cols and '품번' in cols:
                        data_map['po'] = temp_df
                        identified = True; break
                    # 4. 품목정보 (상품코드, B/P무게)
                    elif 'B/P무게' in cols or 'B/P weight' in cols:
                        data_map['itm'] = temp_df
                        identified = True; break
                    # 5. 시판스펙관리 (품번, 4개월판매량)
                    elif '4개월판매량' in cols:
                        data_map['rtl'] = temp_df
                        identified = True; break
                except:
                    continue
    return data_map

# --- 3. 메인 화면 ---

st.title("📊 특판 모양지 통합 오더 관리 대시보드")
st.sidebar.header("📁 데이터 업로드")
uploaded_files = st.sidebar.file_uploader("관련 CSV 파일들을 한꺼번에 선택해 주세요", type="csv", accept_multiple_files=True)

if uploaded_files:
    data = identify_data(uploaded_files)
    
    # 필수 파일(수주, 재고) 확인
    if 'exp' in data and 'stk' in data:
        df_exp = data['exp']
        df_stk = data['stk']
        df_po = data.get('po')
        df_itm = data.get('itm')
        df_rtl = data.get('rtl')
        
        # 컬럼명 표준화 (상품코드/품번 혼용 대응)
        exp_item_col = '상품코드' if '상품코드' in df_exp.columns else '품번'
        
        # 분석 설정
        st.sidebar.divider()
        unit = st.sidebar.radio("🗓️ 분석 단위 선택", ["월별", "분기별"])
        months_to_show = st.sidebar.slider("분석 기간(개월)", 6, 24, 12)
        
        # 데이터 수치화
        df_exp['수주잔량_n'] = to_num_series(df_exp['수주잔량'])
        df_stk['재고수량_n'] = to_num_series(df_stk['재고수량'])
        
        # 평량 매핑
        weight_map = {}
        if df_itm is not None:
            itm_code_col = '상품코드' if '상품코드' in df_itm.columns else '품번'
            w_col = 'B/P무게' if 'B/P무게' in df_itm.columns else 'B/P weight'
            weight_map = df_itm.set_index(itm_code_col)[w_col].to_dict()

        # 분석 대상 품번 (수주잔량이 있는 모든 품번)
        all_items = df_exp[df_exp['수주잔량_n'] > 0][exp_item_col].unique()
        
        # 기간 생성
        start_date = datetime.now().replace(day=1)
        if unit == "월별":
            periods = [(start_date + pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(months_to_show)]
        else:
            # 분기별 (현재 분기부터 시작)
            periods = []
            for i in range(months_to_show // 3):
                target_date = start_date + pd.DateOffset(months=i*3)
                periods.append(f"{target_date.year} Q{(target_date.month-1)//3 + 1}")
        
        matrix_data = []

        for item in all_items:
            # 기본 정보
            item_name = df_exp[df_exp[exp_item_col] == item]['상품명'].iloc[0] if '상품명' in df_exp.columns else "알수없음"
            bw = weight_map.get(item, 70.0)
            bw = float(bw) if str(bw).replace('.','').isdigit() else 70.0
            
            # 1. 초기 가용 재고 (현재고 + PO환산)
            inv_m = to_num_series(df_stk[df_stk['품번'] == item]['재고수량_n']).sum()
            if df_po is not None:
                po_kg = to_num_series(df_po[df_po['품번'] == item]['PO 수량']).sum()
                inv_m += (po_kg * 1000) / (bw * 1.26)
            
            # 2. 시판 월 소요량
            rtl_m = 0
            if df_rtl is not None:
                rtl_m = to_num_series(df_rtl[df_rtl['품번'] == item]['4개월판매량']).sum() / 4
            
            # 3. 특판 수요 배분 (납품예정일 기준)
            item_exp = df_exp[df_exp[exp_item_col] == item].copy()
            item_exp['date'] = pd.to_datetime(item_exp['납품예정일'].astype(str), errors='coerce')
            
            row_cons = {"품번": item, "상품명": item_name, "구분": "예상소요량(m)"}
            row_inv = {"품번": item, "상품명": item_name, "구분": "예상재고량(m)"}
            
            current_running_inv = inv_m
            
            for p in periods:
                if unit == "월별":
                    p_start = datetime.strptime(p, "%Y-%m")
                    p_end = p_start + pd.DateOffset(months=1)
                    spec_m = item_exp[(item_exp['date'] >= p_start) & (item_exp['date'] < p_end)]['수주잔량_n'].sum()
                    total_m = spec_m + rtl_m
                else:
                    # 분기별 합산
                    q_year = int(p.split(' ')[0])
                    q_num = int(p.split('Q')[1])
                    p_start = datetime(q_year, (q_num-1)*3 + 1, 1)
                    p_end = p_start + pd.DateOffset(months=3)
                    spec_m = item_exp[(item_exp['date'] >= p_start) & (item_exp['date'] < p_end)]['수주잔량_n'].sum()
                    total_m = spec_m + (rtl_m * 3)
                
                current_running_inv -= total_m
                row_cons[p] = round(total_m)
                row_inv[p] = round(current_running_inv)
            
            matrix_data.append(row_cons)
            matrix_data.append(row_inv)

        # 결과 테이블화
        result_df = pd.DataFrame(matrix_data)

        # 음영 스타일 함수
        def color_inventory(val):
            if isinstance(val, (int, float, np.integer, np.floating)):
                if val < 0: return 'background-color: #ffcccc; color: #990000; font-weight: bold;' # 재고부족 빨강
                return 'background-color: #e6ffed; color: #006600;' # 재고있음 초록
            return ''

        st.subheader(f"📅 품번별 {unit} 통합 재고 수지 (현재고 + PO 포함)")
        st.dataframe(
            result_df.style.applymap(color_inventory, subset=periods),
            use_container_width=True,
            height=600
        )
        
        st.success("✅ 분석 완료! 빨간색으로 표시된 시점은 재고 부족이 예상되므로 발주가 필요합니다.")
        
        # 다운로드
        csv = result_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 분석 결과 다운로드 (CSV)", csv, f"special_order_report_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

    else:
        st.warning("⚠️ '수주예정등록'과 '현재고' 파일이 인식되지 않았습니다. 파일 안의 컬럼명을 확인해 주세요.")
else:
    st.info("👈 왼쪽 사이드바에 분석할 CSV 파일들을 모두 올려주세요. (파일명은 상관없습니다)")
