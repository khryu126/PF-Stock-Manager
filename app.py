import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- [1. 설정 및 리드타임 마스터] ---
st.set_page_config(page_title="P·Forecast Stock Manager", layout="wide")

LT_MASTER = {
    'SH': 1, 'KD': 2, 'QZ': 2, 'SE': 6, 'SRL': 8, 'SP': 8
}

# --- [2. 데이터 정제 유틸리티] ---
def clean_numeric_data(series):
    """문자열 숫자(콤마 포함)를 실수형으로 변환"""
    if series.dtype == 'object':
        series = series.astype(str).str.replace(',', '').str.replace('"', '').str.strip()
        series = series.replace(['', 'nan', 'None'], np.nan)
    return pd.to_numeric(series, errors='coerce').fillna(0)

def parse_date(series):
    """날짜 형식 표준화"""
    return pd.to_datetime(series, errors='coerce')

def get_pattern_group(df_item, target_id):
    """품번 이원화(Code Chain) 추적"""
    target_id = str(target_id).strip()
    related = {target_id}
    
    # 품목정보에서 연계 품번 탐색
    links = df_item[(df_item['상품코드'] == target_id) | 
                    (df_item['이전상품코드'] == target_id) | 
                    (df_item['변경상품코드'] == target_id)]
    
    for _, row in links.iterrows():
        for col in ['상품코드', '이전상품코드', '변경상품코드']:
            if col in df_item.columns:
                val = str(row[col]).strip()
                if val and val.lower() != 'nan' and val != '0':
                    related.add(val)
    return list(related)

# --- [3. 상세 팝업창 (Drill-down)] ---
@st.dialog("상세 수주 및 납기 현황")
def show_detail_popup(group_ids, df_bl):
    st.write(f"🔎 연계 품번 그룹: {', '.join(group_ids)}")
    detail = df_bl[df_bl['상품코드'].isin(group_ids)].copy()
    
    if detail.empty:
        st.info("현재 수주 잔량이 없습니다.")
        return

    today = datetime.now()
    detail['상태'] = detail['납품예정일'].apply(lambda x: "⚠️ 납기경과" if pd.notnull(x) and x < today else "정상")
    
    cols = ['상태', '현장명', '건설사', '수주잔량', '납품예정일', '메모']
    # 존재하는 컬럼만 출력
    actual_cols = [c for c in cols if c in detail.columns]
    st.dataframe(detail[actual_cols].sort_values('납품예정일'), use_container_width=True, hide_index=True)

# --- [4. 메인 앱 UI] ---
st.title("📦 P·Forecast Stock Manager")
st.caption("건설 특판 모양지 통합 오더 및 재고 수지 관리 시스템")

# 파일 업로드 섹션
uploaded_files = st.sidebar.file_uploader("5종의 CSV 파일을 모두 선택하세요", accept_multiple_files=True)

data = {}
if uploaded_files:
    for f in uploaded_files:
        # [핵심 수정] 인코딩 에러 방지 로직
        try:
            # 먼저 UTF-8로 시도
            df = pd.read_csv(f, encoding='utf-8')
        except UnicodeDecodeError:
            # 실패하면 한국어 전용 인코딩(CP949)으로 재시도
            f.seek(0) # 파일 읽기 위치 초기화
            df = pd.read_csv(f, encoding='cp949')
        
        df.columns = [str(c).strip() for c in df.columns]
        
        # 파일 자동 판별
        cols_text = "".join(df.columns)
        if "수주잔량" in cols_text: data['backlog'] = df
        elif "PO" in cols_text or "미선적" in cols_text: data['po'] = df
        elif "현재고" in cols_text or "재고수량" in cols_text: data['stock'] = df
        elif "시판" in cols_text: data['retail'] = df
        elif "최종생산지" in cols_text or "상품명" in cols_text: data['item'] = df

# 데이터 처리 및 시각화
if len(data) >= 5:
    # 데이터 로드
    df_item, df_bl, df_po, df_st, df_retail = data['item'], data['backlog'], data['po'], data['stock'], data['retail']

    # 모든 숫자형 컬럼 정제
    for df in [df_bl, df_po, df_st, df_retail]:
        for col in df.columns:
            if any(k in col for k in ['잔량', '수량', '현재고', 'weight', '평량']):
                df[col] = clean_numeric_data(df[col])

    df_bl['납품예정일'] = parse_date(df_bl['납품예정일'])
    df_po['입고요청일'] = parse_date(df_po.get('입고요청일', df_po.get('PO일자'))) # 날짜 컬럼 유연화

    # 타임라인 설정
    today_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    month_range = [today_start + pd.DateOffset(months=i) for i in range(12)]
    month_cols = [m.strftime('%Y-%m') for m in month_range]

    target_ids = df_bl[df_bl['수주잔량'] > 0]['상품코드'].unique()
    matrix_rows = []
    processed_groups = set()

    for pid in target_ids:
        group = sorted(get_pattern_group(df_item, pid))
        group_key = tuple(group)
        if group_key in processed_groups: continue
        processed_groups.add(group_key)

        # 기초 정보 추출
        relevant_items = df_item[df_item['상품코드'].isin(group)]
        item_info = relevant_items.iloc[0] if not relevant_items.empty else {}
        site_code = str(item_info.get('최종생산지명', item_info.get('최종생산지', 'ETC')))
        lt = LT_MASTER.get(site_code, 0)
        
        is_retail = "🏷️" if any(str(g) in df_retail.iloc[:, 8].astype(str).values for g in group) else ""
        has_chain = "🔄" if len(group) > 1 else ""
        
        # 재고 수지 전개
        curr_stock = df_st[df_st.iloc[:, 7].isin(group)].iloc[:, 17].sum() if 'stock' in data else 0
        overdue_demand = df_bl[(df_bl['상품코드'].isin(group)) & (df_bl['납품예정일'] < today_start)]['수주잔량'].sum()
        
        running_inv = curr_stock - overdue_demand
        row_demand = {"납기경과": overdue_demand}
        row_stock = {"납기경과": running_inv}

        for m_date in month_range:
            m_str = m_date.strftime('%Y-%m')
            m_dem = df_bl[(df_bl['상품코드'].isin(group)) & (df_bl['납품예정일'] >= m_date) & (df_bl['납품예정일'] < m_date + pd.DateOffset(months=1))]['수주잔량'].sum()
            
            m_po_data = df_po[(df_po.iloc[:, 12].isin(group)) & (parse_date(df_po['입고요청일']) >= m_date) & (parse_date(df_po['입고요청일']) < m_date + pd.DateOffset(months=1))]
            m_sup = 0
            for _, r in m_po_data.iterrows():
                bw = clean_numeric_data(pd.Series([r.get('B/P weight', 70)]))[0]
                m_sup += (clean_numeric_data(pd.Series([r.get('PO잔량(미선적)', 0)]))[0] * 1000) / ((bw if bw > 0 else 70) * 1.26)
            
            running_inv = (running_inv + m_sup) - m_dem
            row_demand[m_str] = round(m_dem, 0)
            row_stock[m_str] = round(running_inv, 0)

        title = f"{pid} {is_retail}{has_chain}{'⚠️' if overdue_demand > 0 else ''}"
        common = {"품번": title, "생산지(LT)": f"{site_code}({lt}M)", "group": group}
        matrix_rows.append({**common, "구분": "소요량(m)", **row_demand})
        matrix_rows.append({**common, "구분": "예상재고(m)", **row_stock})

    if matrix_rows:
        result_df = pd.DataFrame(matrix_rows)
        
        def style_matrix(row):
            styles = [''] * len(row)
            if row['구분'] == "예상재고(m)":
                lt_val = int(row['생산지(LT)'].split('(')[1].replace('M)', ''))
                for i, col in enumerate(row.index):
                    if col == "납기경과" and row[col] < 0:
                        styles[i] = 'background-color: #9e0000; color: white'
                    elif '-' in col and row[col] < 0:
                        col_dt = datetime.strptime(col, '%Y-%m')
                        limit_dt = today_start + pd.DateOffset(months=lt_val)
                        styles[i] = 'background-color: #ff4b4b; color: white' if col_dt <= limit_dt else 'background-color: #ffeb3b; color: black'
            return styles

        st.subheader("📊 통합 수급 분석 매트릭스")
        selection = st.dataframe(
            result_df.style.apply(style_matrix, axis=1),
            use_container_width=True, hide_index=True,
            column_order=["품번", "생산지(LT)", "구분", "납기경과"] + month_cols,
            on_select="rerun", selection_mode="single_row"
        )

        if selection.selection.rows:
            sel_idx = selection.selection.rows[0]
            if st.button(f"🔍 {result_df.iloc[sel_idx]['품번']} 상세 내역 보기"):
                show_detail_popup(result_df.iloc[sel_idx]['group'], df_bl)
else:
    st.warning("왼쪽 사이드바에서 5종의 CSV 파일을 업로드해 주세요.")
