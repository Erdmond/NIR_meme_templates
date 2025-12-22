import io
from PIL import Image
import streamlit as st
from src import MemeSearchEngine

st.set_page_config("🔍 Поиск мемов по смыслу", "🤖", "wide")

@st.cache_resource
def load_engine():
    return MemeSearchEngine('data/memes_post.parquet')

def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subheader {
        font-size: 1rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 1rem;
    }
    .meme-container {
        margin-bottom: 1.5rem;
    }
    .score-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-top: 0.3rem;
    }
    .high-score { background-color: #D1FAE5; color: #065F46; }
    .medium-score { background-color: #FEF3C7; color: #92400E; }
    .low-score { background-color: #FEE2E2; color: #991B1B; }
    </style>
    """, unsafe_allow_html=True)

def get_score_badge_class(score):
    if score >= 0.7:
        return "high-score"
    elif score >= 0.4:
        return "medium-score"
    else:
        return "low-score"

def main():
    load_css()
    
    st.markdown('<h1 class="main-header">🔍 Поиск мемов по смыслу</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Опишите картинку — найдем подходящий шаблон мема</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Параметры поиска")
        
        search_mode = st.radio(
            "Режим поиска:",
            ["По количеству", "По схожести", "Гибридный"],
            index=0
        )
        
        if search_mode == "По количеству":
            top_k = st.slider("Количество результатов", 1, 15, 6)
            min_similarity = 0.0
        elif search_mode == "По схожести":
            min_similarity = st.slider("Минимальная схожесть", 0.0, 1.0, 0.3, 0.05)
            top_k = 1000
        else:
            top_k = st.slider("Максимальное количество", 1, 15, 6)
            min_similarity = st.slider("Минимальная схожесть", 0.0, 1.0, 0.3, 0.05)
        
        st.divider()
        st.subheader("Отображение")
        layout_cols = st.selectbox("Колонок в строке", [2, 3, 4], index=1)
        
        st.divider()
        st.subheader("ℹ️ О системе")
        st.markdown("""
        **Технологии:**
        - Модель: `paraphrase-multilingual-MiniLM-L12-v2`
        - Поиск: косинусная схожесть
        - База: 2.3k шаблонов мемов
        
        **Как работает:**
        1. Ваш запрос переводится в вектор
        2. Ищутся близкие векторы английских мемов
        3. Возвращаются наиболее релевантные
        
        **Особенности:**
        - Кросс-языковой поиск (русский → английский)
        - Семантическое понимание смысла
        - 3 режима поиска
        """)
    
    engine = load_engine()

    with st.form(key='search_form'):
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "Введите запрос:",
                placeholder="Например: 'Котик грустит'",
                key='query_input'
            )
        with col2:
            st.write("")
            st.write("")
            search_clicked = st.form_submit_button(
                "Искать", 
                use_container_width=True, 
                type="primary"
            )

    if search_clicked and query:
        with st.spinner("Ищем..."):
            results = engine.search(query, top_k=top_k, min_similarity=min_similarity)
        
        st.session_state.results = results
        st.session_state.last_query = query
        st.session_state.layout_cols = layout_cols
    
    if hasattr(st.session_state, 'results') and not st.session_state.results.empty:
        results = st.session_state.results
        query = st.session_state.last_query
        layout_cols = st.session_state.get('layout_cols', 3)
        
        st.write("---")
        cols = st.columns(4)
        with cols[0]:
            st.metric("Найдено", len(results))
        with cols[1]:
            avg_score = results['score'].mean()
            st.metric("Средняя схожесть", f"{avg_score:.3f}")
        with cols[2]:
            st.metric("Максимум", f"{results['score'].max():.3f}")
        with cols[3]:
            st.metric("Минимум", f"{results['score'].min():.3f}")
        
        st.write(f"### Результаты для: '{query}'")
        
        results_list = list(results.iterrows())
        
        for i in range(0, len(results_list), layout_cols):
            cols = st.columns(layout_cols)
            
            for col_idx in range(layout_cols):
                if i + col_idx < len(results_list):
                    idx, row = results_list[i + col_idx]
                    
                    with cols[col_idx]:
                        st.markdown('<div class="meme-container">', unsafe_allow_html=True)
                        
                        try:
                            image = Image.open(io.BytesIO(row['local_path']))
                            st.image(image, use_container_width=True)
                            
                            st.markdown(f"**{row['name']}**")
                            
                            badge_class = get_score_badge_class(row['score'])
                            st.markdown(
                                f'<div class="score-badge {badge_class}">'
                                f'Схожесть: {row["score"]:.3f}'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        except Exception as e:
                            st.error(f"Ошибка загрузки: {str(e)[:50]}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🔄 Новый поиск", use_container_width=True):
            st.session_state.pop('results', None)
            st.session_state.pop('last_query', None)
            st.rerun()
    
    elif hasattr(st.session_state, 'results') and st.session_state.results.empty:
        st.warning("По вашему запросу ничего не найдено.")
        
        with st.expander("Примеры запросов"):
            st.write("- **грустный кот** → sad cat")
            st.write("- **радость победы** → success kid")
            st.write("- **удивление** → surprised pikachu")
            st.write("- **работа за компьютером** → programmer")
            st.write("- **усталость** → tired")
            st.write("- **смешная ситуация** → funny situation")
    
    st.write("---")
    st.caption(f"© 2025 NIR Meme Search • OmSTU")

if __name__ == "__main__":
    main()
