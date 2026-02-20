import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- [설정 및 마스터 데이터] ---
st.set_page_config(page_title="P·Forecast Stock Manager", layout="wide")

# 생산지별 리드타임(LT) 매핑
LT_MASTER = {
    'SH': 1, 'KD': 2, 'QZ': 2, 'SE': 6, 'SRL': 8, 'SP': 8
}

# --- [유틸리티 함수] ---
def clean_numeric(series):
    """문자열 내 콤마 제거 및 실수 변환, NaN 처리"""
    if series.dtype == 'object':
        series = series.str.replace(',', '').str.strip()
    return pd.to_numeric(series, errors='coerce').fillna(0)

def parse_date(series):
    """다양한 날짜 형식 대응 (YYYYMMDD, YYYY-MM-DD 등)"""
    return pd.to_datetime(series, errors='coerce')

def get_pattern_group(df_item, target_id):
    """이전/변경 품번을 추적하여 연계된 모든 품번 리스트 반환 (Code Chain)"""
    target_id = str(target_id).strip()
    related = {target_id}
    
    # 1단계 연결 찾기
    links = df_item[(df_item['상품코드'] == target_id) | 
                    (df_item['이전상품코드'] == target_id) | 
                    (df_item['변경상품코드'] == target_id)]
    
    for _, row in links.iterrows():
        for col in ['상품코드', '이전상품코드', '변경상품코드']:
            val = str(row[col]).strip()
            if val and val != 'nan' and val != '0':
                related.add(val)
    return list(related)

# --- [메인 로직: 상세 팝업창] ---
@st.dialog("현장별 수주 상세 내역 (Drill-down)")
def show_detail_dialog(group_ids, df_bl):
    st.write(f"🔍 분석 품번 그룹: {', '.join(group_ids)}")
    
    # 해당 그룹의 수주 데이터 추출
    detail = df_bl[df_bl['상품코드'].isin(group_ids)].copy()
    
    if detail.empty:
        st.info("해당 품번의 수주 잔량 데이터가 없습니다.")
        return

    # 납기 상태 구분
    today = datetime.now()
    detail['상태'] = detail['납품예정일'].apply(lambda x: "⚠️ 납기경과" if x < today else "정상")
    
    # 출력용 정리
    display_cols = ['상태', '현장명', '건설사', '수주잔량', '납품예정일', '메모']
    st.dataframe(detail[display_cols].sort_values('납품예정일'), use_container_width=True, hide_index=True)
    st.caption("※ 납기경과 물량은 유령 잔량 여부를 현업 담당자와 확인하시기 바랍니다.")

# --- [앱 UI 시작] ---
st.title("📦 P·Forecast Stock Manager")
st.markdown("##### 건설 특판 모양지 통합 수급 예측 시스템")

# 1. 데이터 업로드 및 전처리
uploaded_files = st.sidebar.file_uploader("5종 CSV 업로드 (품목, 시판, 수주, PO, 재고)", accept_multiple_files=True)

data = {}
if uploaded_files:
    for f in uploaded_files:
        df = pd.read_csv(f).rename(columns=lambda x: x.strip())
        # 간단한 키워드 기반 파일 매핑
        cols = "".join(df.columns)
        if "수주잔량" in cols: data['backlog'] = df
        elif "PO" in cols or "미선적" in cols: data['po'] = df
        elif "현재고" in cols: data['stock'] = df
        elif "시판" in cols: data['retail'] = df
        elif "생산지" in cols and "상품코드" in cols: data['item'] = df

# 필수 파일 체크
if len(data) >= 5:
    # 데이터 표준화
    df_item = data['item']
    df_bl = data['backlog']
    df_po = data['po']
    df_st = data['stock']
    df_retail = data['retail']

    # 수주 데이터 정제 (잔량 > 0 만)
    df_bl['수주잔량'] = clean_numeric(df_bl['수주잔량'])
    df_bl['납품예정일'] = parse_date(df_bl['납품예정일'])
    df_bl = df_bl[df_bl['수주잔량'] > 0]

    # 분석 대상 품번 리스트 (수주잔고 있는 것들)
    target_ids = df_bl['상품코드'].unique()
    
    # 타임라인 설정
    today = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    months = [today + pd.DateOffset(months=i) for i in range(12)]
    month_cols = [m.strftime('%Y-%m') for m in months]
    
    matrix_data = []
    processed_groups = set()

    for pid in target_ids:
        # 그룹화 (Code Chain)
        group = sorted(get_pattern_group(df_item, pid))
        group_key = tuple(group)
        if group_key in processed_groups: continue
        processed_groups.add(group_key)

        # 기초 정보
        item_info = df_item[df_item['상품코드'].isin(group)].iloc[0]
        site_code = str(item_info.get('최종생산지명', 'ETC'))
        lt = LT_MASTER.get(site_code, 0)
        
        # 태그 생성
        is_retail = "🏷️" if any(str(g) in df_retail['품번'].astype(str).values for g in group) else ""
        has_chain = "🔄" if len(group) > 1 else ""
        
        # 현재고 (그룹 합산)
        curr_stock = clean_numeric(df_st[df_st['품번'].isin(group)]['재고수량']).sum()
        
        # 1. 납기경과 소요량 계산
        overdue_demand = df_bl[(df_bl['상품코드'].isin(group)) & (df_bl['납품예정일'] < today)]['수주잔량'].sum()
        has_overdue = "⚠️" if overdue_demand > 0 else ""

        # 2. 타임라인 수급 전개
        running_inv = curr_stock - overdue_demand
        row_demands = {"납기경과": overdue_demand}
        row_stocks = {"납기경과": running_inv}

        for m_date in months:
            m_str = m_date.strftime('%Y-%m')
            # 해당 월 소요
            m_demand = df_bl[(df_bl['상품코드'].isin(group)) & 
                             (df_bl['납품예정일'] >= m_date) & 
                             (df_bl['납품예정일'] < m_date + pd.DateOffset(months=1))]['수주잔량'].sum()
            
            # 해당 월 입고 (PO 단위환산)
            m_po = df_po[(df_po['품번'].isin(group)) & (parse_date(df_po['입고요청일']) >= m_date) & 
                         (parse_date(df_po['입고요청일']) < m_date + pd.DateOffset(months=1))]
            
            m_supply = 0
            for _, r in m_po.iterrows():
                bw = clean_numeric(pd.Series([r.get('B/P weight', 70)]))[0]
                bw = 70 if bw == 0 else bw
                m_supply += (clean_numeric(pd.Series([r.get('PO잔량(미선적)', 0)]))[0] * 1000) / (bw * 1.26)
            
            running_inv = (running_inv + m_supply) - m_demand
            row_demands[m_str] = round(m_demand, 0)
            row_stocks[m_str] = round(running_inv, 0)

        # 매트릭스 행 추가
        base_info = {"품번": f"{pid} {is_retail}{has_chain}{has_overdue}", "생산지(LT)": f"{site_code}({lt}M)"}
        matrix_data.append({**base_info, "구분": "소요량(m)", **row_demands, "group": group})
        matrix_data.append({**base_info, "구분": "예상재고(m)", **row_stocks, "group": group})

    # 데이터프레임 생성 및 출력
    res_df = pd.DataFrame(matrix_data)
    
    # 스타일링 함수
    def apply_style(row):
        styles = [''] * len(row)
        if row['구분'] == "예상재고(m)":
            lt_val = int(row['생산지(LT)'].split('(')[1].replace('M)', ''))
            for i, col in enumerate(row.index):
                if col == "납기경과" and row[col] < 0:
                    styles[i] = 'background-color: #9e0000; color: white' # 심각한 과부족
                elif '-' in col and row[col] < 0:
                    col_date = datetime.strptime(col, '%Y-%m')
                    limit_date = today + pd.DateOffset(months=lt_val)
                    if col_date <= limit_date:
                        styles[i] = 'background-color: #ff4b4b; color: white' # LT 내 고갈
                    else:
                        styles[i] = 'background-color: #ffeb3b' # LT 외 고갈
        return styles

    st.subheader("📊 통합 수급 분석 매트릭스")
    st.info("💡 행을 클릭한 후 하단의 '상세보기' 버튼을 누르면 현장별 납기 내역을 확인할 수 있습니다.")
    
    # 데이터프레임 표시 (선택 가능)
    event = st.dataframe(
        res_df.style.apply(apply_style, axis=1),
        use_container_width=True,
        hide_index=True,
        column_order=["품번", "생산지(LT)", "구분", "납기경과"] + month_cols,
        on_select="rerun",
        selection_mode="single_row"
    )

    # 선택된 행의 상세 팝업 호출
    if len(event.selection.rows) > 0:
        selected_idx = event.selection.rows[0]
        selected_group = res_df.iloc[selected_idx]['group']
        if st.button(f"🔍 {res_df.iloc[selected_idx]['품번']} 상세 내역 보기"):
            show_detail_dialog(selected_group, df_bl)

else:
    st.warning("분석을 위해 5종의 CSV 파일을 측면 바에 업로드해 주세요.")
    st.markdown("""
    **필수 파일 정보:**
    1. **품목정보:** 상품코드, 이전/변경코드, 최종생산지명 포함
    2. **수주예정등록:** 상품코드, 수주잔량, 납품예정일 포함
    3. **PO:** 품번, PO잔량(미선적), B/P weight, 입고요청일 포함
    4. **현재고:** 품번, 재고수량 포함
    5. **시판스펙관리:** 품번 포함 (태그 표시용)
    """)
