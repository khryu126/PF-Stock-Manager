import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --- [1. 기본 설정 및 마스터 데이터] ---
st.set_page_config(page_title="P·Forecast Stock Manager", layout="wide")

LT_MASTER = {
    'SH': 1, 'KD': 2, 'QZ': 2, 'SE': 6, 'SRL': 8, 'SP': 8
}

# --- [2. 데이터 정제 및 로드 유틸리티] ---
def clean_numeric(series):
    if series.dtype == 'object':
        series = series.astype(str).str.replace(',', '').str.replace('"', '').str.strip()
        series = series.replace(['', 'nan', 'None'], np.nan)
    return pd.to_numeric(series, errors='coerce').fillna(0)

def parse_date(series):
    return pd.to_datetime(series, errors='coerce')

def smart_load_csv(file):
    """빈 줄 건너뛰기 및 다중 인코딩 지원 지능형 로더"""
    try:
        # 한국어 엑셀 CSV 전용 인코딩 리스트
        encodings = ['cp949', 'utf-8-sig', 'utf-8', 'euc-kr']
        for enc in encodings:
            try:
                file.seek(0)
                df = pd.read_csv(file, encoding=enc)
                # 데이터가 너무 적거나 제목이 Unnamed로만 되어 있으면 한 줄 아래부터 다시 시도
                if df.columns.str.contains('Unnamed').sum() > len(df.columns) * 0.5:
                    for i in range(1, 4):
                        file.seek(0)
                        df = pd.read_csv(file, skiprows=i, encoding=enc)
                        if not df.columns.str.contains('Unnamed').all():
                            break
                return df
            except:
                continue
        return None
    except Exception as e:
        return None

def get_pattern_group(df_item, target_id):
    target_id = str(target_id).strip()
    related = {target_id}
    if df_item is not None:
        # '상품코드', '이전상품코드', '변경상품코드' 컬럼 중 존재하는 것만 활용
        search_cols = [c for c in ['상품코드', '이전상품코드', '변경상품코드'] if c in df_item.columns]
        if search_cols:
            query = " | ".join([f"(`{c}` == '{target_id}')" for c in search_cols])
            try:
                links = df_item.query(query)
                for _, row in links.iterrows():
                    for col in search_cols:
                        val = str(row[col]).strip()
                        if val and val.lower() != 'nan' and val != '0':
                            related.add(val)
            except:
                pass
    return list(related)

# --- [3. 상세 팝업창] ---
@st.dialog("현장별 수주 상세 내역")
def show_detail_dialog(group_ids, df_bl):
    st.write(f"🔍 분석 품번 그룹: {', '.join(group_ids)}")
    detail = df_bl[df_bl['상품코드'].isin(group_ids)].copy()
    if detail.empty:
        st.info("수주 데이터가 없습니다.")
        return
    today = datetime.now()
    detail['상태'] = detail['납품예정일'].apply(lambda x: "⚠️ 납기경과" if pd.notnull(x) and x < today else "정상")
    cols = ['상태', '현장명', '건설사', '수주잔량', '납품예정일', '메모']
    actual_cols = [c for c in cols if c in detail.columns]
    st.dataframe(detail[actual_cols].sort_values('납품예정일'), use_container_width=True, hide_index=True)

# --- [4. 메인 UI 및 파일 인식 로직] ---
st.title("📦 P·Forecast Stock Manager")
st.caption("특판 모양지 통합 수급 예측 시스템 (리드타임 및 품번 연계 대응)")

uploaded_files = st.sidebar.file_uploader("5종의 CSV 파일을 한꺼번에 선택하세요", accept_multiple_files=True)

data = {}
# 파일 판별을 위한 복합 키워드 사전
RECOGNITION_MAP = {
    "backlog": {"name": "수주예정(Demand)", "keys": ["수주잔량", "현장명", "총예상수량"], "found": False},
    "po": {"name": "구매발주(PO)", "keys": ["PO잔량", "미선적", "B/P weight"], "found": False},
    "stock": {"name": "현재고(Stock)", "keys": ["재고수량", "현재고액", "본사창고"], "found": False},
    "item": {"name": "품목정보(Master)", "keys": ["최종생산지", "이전상품코드", "변경상품코드"], "found": False},
    "retail": {"name": "시판스펙(Retail)", "keys": ["출시예정", "4개월판매량", "제시단가", "시판"], "found": False}
}

unrecognized_files = []

if uploaded_files:
    for f in uploaded_files:
        df = smart_load_csv(f)
        if df is not None:
            df.columns = [str(c).strip() for c in df.columns]
            cols_text = "|".join(df.columns)
            
            matched = False
            for file_id, info in RECOGNITION_MAP.items():
                if any(k in cols_text for k in info["keys"]):
                    data[file_id] = df
                    RECOGNITION_MAP[file_id]["found"] = True
                    matched = True
                    break
            if not matched:
                unrecognized_files.append({"filename": f.name, "columns": df.columns.tolist()})

# 사이드바 로드 상태 및 디버깅 정보
st.sidebar.markdown("---")
st.sidebar.subheader("📁 데이터 로드 상태")
for k, v in RECOGNITION_MAP.items():
    if v["found"]: st.sidebar.success(f"✅ {v['name']}")
    else: st.sidebar.error(f"❌ {v['name']} (미인식)")

if unrecognized_files:
    with st.sidebar.expander("⚠️ 미인식 파일 컬럼 확인"):
        for f in unrecognized_files:
            st.text(f"파일: {f['filename']}")
            st.caption(f"감지된 컬럼: {', '.join(f['columns'][:5])}...")

# 분석 시작
if len(data) >= 5:
    st.success("✅ 모든 파일이 인식되었습니다. 분석 매트릭스를 생성합니다.")
    df_item, df_bl, df_po, df_st, df_retail = data['item'], data['backlog'], data['po'], data['stock'], data['retail']

    # 데이터 정제
    for df_key in ['backlog', 'po', 'stock', 'retail']:
        df = data[df_key]
        for col in df.columns:
            if any(k in col for k in ['잔량', '수량', '현재고', 'weight', '평량', '판매량']):
                df[col] = clean_numeric(df[col])
    
    df_bl['납품예정일'] = parse_date(df_bl['납품예정일'])
    df_po['입고요청일'] = parse_date(df_po.get('입고요청일', df_po.get('PO일자')))

    # 타임라인 설정
    today_base = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    months = [today_base + pd.DateOffset(months=i) for i in range(12)]
    month_cols = [m.strftime('%Y-%m') for m in months]

    # 분석 대상 (수주잔량 있는 품번)
    target_ids = df_bl[df_bl['수주잔량'] > 0]['상품코드'].unique()
    matrix_rows = []
    processed_groups = set()

    for pid in target_ids:
        group = sorted(get_pattern_group(df_item, pid))
        group_key = tuple(group)
        if group_key in processed_groups: continue
        processed_groups.add(group_key)

        # 기초 정보
        item_rows = df_item[df_item['상품코드'].isin(group)]
        item_info = item_rows.iloc[0] if not item_rows.empty else {}
        site_code = str(item_info.get('최종생산지명', item_info.get('최종생산지', 'ETC')))
        lt = LT_MASTER.get(site_code, 0)
        
        # 시판/연계 태그 (컬럼 위치 기반 유연화)
        is_retail = "🏷️" if any(str(g) in df_retail.iloc[:, 8].astype(str).values for g in group) else ""
        has_chain = "🔄" if len(group) > 1 else ""
        
        # 수지 계산
        # 현재고 (현재고 컬럼 또는 수량 컬럼 합산)
        st_cols = [c for c in df_st.columns if '재고수량' in c or '현재고' in c]
        total_stk = df_st[df_st.get('품번', df_st.columns[0]).isin(group)][st_cols[0]].sum() if st_cols else 0
        
        overdue_dem = df_bl[(df_bl['상품코드'].isin(group)) & (df_bl['납품예정일'] < today_base)]['수주잔량'].sum()
        
        running_inv = total_stk - overdue_dem
        row_dem = {"납기경과": overdue_dem}
        row_stk = {"납기경과": running_inv}

        for m_date in months:
            m_str = m_date.strftime('%Y-%m')
            # 소요량
            m_d = df_bl[(df_bl['상품코드'].isin(group)) & (df_bl['납품예정일'] >= m_date) & (df_bl['납품예정일'] < m_date + pd.DateOffset(months=1))]['수주잔량'].sum()
            # 입고량
            m_p = df_po[(df_po.get('품번', df_po.columns[0]).isin(group)) & (df_po['입고요청일'] >= m_date) & (df_po['입고요청일'] < m_date + pd.DateOffset(months=1))]
            m_s = 0
            for _, r in m_p.iterrows():
                bw = clean_numeric(pd.Series([r.get('B/P weight', 70)]))[0]
                m_s += (clean_numeric(pd.Series([r.get('PO잔량(미선적)', 0)]))[0] * 1000) / ((bw if bw > 0 else 70) * 1.26)
            
            running_inv = (running_inv + m_s) - m_d
            row_dem[m_str] = round(m_d, 0)
            row_stk[m_str] = round(running_inv, 0)

        title = f"{pid} {is_retail}{has_chain}{'⚠️' if overdue_dem > 0 else ''}"
        common = {"품번": title, "생산지(LT)": f"{site_code}({lt}M)", "group": group}
        matrix_rows.append({**common, "구분": "소요량(m)", **row_dem})
        matrix_rows.append({**common, "구분": "예상재고(m)", **row_stk})

    if matrix_rows:
        res_df = pd.DataFrame(matrix_rows)
        
        def style_matrix(row):
            styles = [''] * len(row)
            if row['구분'] == "예상재고(m)":
                try:
                    lt_val = int(row['생산지(LT)'].split('(')[1].replace('M)', ''))
                except: lt_val = 0
                for i, col in enumerate(row.index):
                    if col == "납기경과" and row[col] < 0:
                        styles[i] = 'background-color: #9e0000; color: white'
                    elif '-' in col and row[col] < 0:
                        col_dt = datetime.strptime(col, '%Y-%m')
                        limit_dt = today_base + pd.DateOffset(months=lt_val)
                        styles[i] = 'background-color: #ff4b4b; color: white' if col_dt <= limit_dt else 'background-color: #ffeb3b; color: black'
            return styles

        selection = st.dataframe(
            res_df.style.apply(style_matrix, axis=1),
            use_container_width=True, hide_index=True,
            column_order=["품번", "생산지(LT)", "구분", "납기경과"] + month_cols,
            on_select="rerun", selection_mode="single_row"
        )

        if selection.selection.rows:
            sel_idx = selection.selection.rows[0]
            if st.button(f"🔍 {res_df.iloc[sel_idx]['품번']} 상세 정보 팝업"):
                show_detail_dialog(res_df.iloc[sel_idx]['group'], df_bl)
else:
    st.info("사이드바에 5종의 파일을 모두 업로드해 주세요. 미인식 시 사이드바의 '컬럼 확인'을 참조하세요.")
