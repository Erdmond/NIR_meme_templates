import io
import re
import pymorphy3
import unicodedata
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class MemeSearchPreprocessor:
    """
    Интеллектуальный препроцессор для поиска мемов.
    Адаптирован для модели paraphrase-multilingual-MiniLM-L12-v2.
    """
    
    def __init__(self, use_lemmatization: bool = True, fix_typos: bool = True):
        """
        Args:
            use_lemmatization: применять ли лемматизацию
            fix_typos: исправлять ли частые опечатки
        """
        self.use_lemmatization = use_lemmatization
        self.fix_typos = fix_typos
        
        if self.use_lemmatization:
            self.morph = pymorphy3.MorphAnalyzer()
        
        self.typo_dict = {
            'штож': 'что ж', 'чё': 'что', 'щас': 'сейчас', 'ща': 'сейчас',
            'спс': 'спасибо', 'плиз': 'пожалуйста', 'ок': 'окей',
            'пасиб': 'спасибо', 'прив': 'привет', 'пж': 'пожалуйста',
            'пжл': 'пожалуйста', 'пжлст': 'пожалуйста',
            'руддщ': 'привет', 'рудз': 'привет', 'пф': 'ап',
            'зщ': 'яи', 'щт': 'шт',
            'имхо': 'по моему мнению', 'лол': 'смешно', 'кек': 'смешно',
            'рофл': 'очень смешно', 'омг': 'о боже', 'нн': 'нормально',
            'хз': 'не знаю', 'изи': 'легко', 'гг': 'хорошая игра',
            'впн': 'vpn', 'ид': 'идентификатор',
            'cry': 'плакать', 'lol': 'смеяться', 'omg': 'о боже',
            'wtf': 'что за черт', 'brb': 'скоро вернусь', 'idk': 'не знаю',
            'симпотный': 'симпатичный', 'зделать': 'сделать',
            'вообщем': 'в общем', 'ихний': 'их', 'ложить': 'класть',
            'ездить': 'ехать', 'координально': 'кардинально',
        }
        
        self.do_not_lemmatize = {
            'догe', 'котэ', 'пёсель', 'котейка', 'псина',
            'превед', 'медвед', 'жожык', 'кросавчег',
            'аниме', 'мем', 'гиф', 'стрим', 'стример',
            'ютуб', 'тикток', 'инстаграм',
            'путин', 'трамп', 'байден', 'маск', 'обнима',
            'пепе', 'доге', 'дож', 'жож',
        }
    
    def normalize_text(self, text: str) -> str:
        """Базовая нормализация текста"""
        if not text or not isinstance(text, str):
            return ""
        
        text = unicodedata.normalize('NFKC', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_special_chars(self, text: str, keep_hashtags: bool = True) -> str:
        """
        Удаляет специальные символы, но сохраняет смысл.
        
        Args:
            keep_hashtags: сохранять ли хэштеги
        """
        if keep_hashtags:
            hashtags = re.findall(r'#\w+', text)
            text = re.sub(r'#\w+', ' HASHTAG_PLACEHOLDER ', text)
        
        text = re.sub(r'https?://\S+|www\.\S+', ' URL_PLACEHOLDER ', text)
        text = re.sub(r'\S+@\S+', ' EMAIL_PLACEHOLDER ', text)
        
        smileys = re.findall(r'[:;=]-?[\)\(/\\\]\[DPp]', text)
        text = re.sub(r'[:;=]-?[\)\(/\\\]\[DPp]', ' SMILEY_PLACEHOLDER ', text)
        
        text = re.sub(r'[^\w\s\-\'.,!?]', ' ', text)
        
        if keep_hashtags and hashtags:
            for ht in hashtags:
                text = text.replace('HASHTAG_PLACEHOLDER', ht.lower(), 1)
        
        if smileys:
            for sm in smileys:
                text = text.replace('SMILEY_PLACEHOLDER', sm, 1)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def fix_common_typos(self, text: str) -> str:
        """Исправляет частые опечатки"""
        if not self.fix_typos:
            return text
        
        words = text.split()
        corrected_words = []
        
        for word in words:
            if word in self.typo_dict:
                corrected_words.append(self.typo_dict[word])
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def smart_lemmatization(self, text: str) -> str:
        """
        Умная лемматизация с исключениями для мем-культуры.
        """
        if not self.use_lemmatization or not hasattr(self, 'morph'):
            return text
        
        words = text.split()
        lemmatized_words = []
        
        for word in words:
            if any(ph in word for ph in ['PLACEHOLDER', 'URL', 'EMAIL', 'HASHTAG', 'SMILEY']):
                lemmatized_words.append(word)
                continue
            
            if word.lower() in self.do_not_lemmatize:
                lemmatized_words.append(word)
                continue
            
            if re.match(r'^[a-zA-Z]+$', word):
                lemmatized_words.append(word.lower())
                continue
            
            try:
                parsed = self.morph.parse(word)[0]
                lemma = parsed.normal_form
                
                if word[0].isupper() and len(word) > 1:
                    lemma = lemma.capitalize()
                
                lemmatized_words.append(lemma)
            except:
                lemmatized_words.append(word)
        
        return ' '.join(lemmatized_words)
    
    def preprocess(self, text: str, for_search: bool = True) -> str:
        """
        Основной метод предобработки.
        
        Args:
            text: входной текст
            for_search: True если это поисковый запрос, False если это текст мема
        """
        if not text:
            return ""
        
        text = self.normalize_text(text)
        
        if self.fix_typos:
            text = self.fix_common_typos(text)
        
        text = self.remove_special_chars(text, keep_hashtags=not for_search)
        
        if self.use_lemmatization:
            text = self.smart_lemmatization(text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        words = text.split()
        if len(words) > 50:
            text = ' '.join(words[:50])
        
        return text
    
    def preprocess_batch(self, texts: list[str], for_search: bool = True) -> list[str]:
        """Пакетная обработка текстов"""
        return [self.preprocess(text, for_search) for text in texts]


class MemeSearchEngine:
    def __init__(self, data_path: str, model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Инициализация поискового движка.
        
        Args:
            data_path: путь к файлу с датасетом и эмбеддингами (.parquet)
            model_name: название модели для эмбеддингов
        """
        self.df = pd.read_parquet(data_path)
        self.model = SentenceTransformer(model_name)
        self.meme_embeddings = np.array(self.df['embedding'].tolist())
        self.preprocessor = MemeSearchPreprocessor()
    
    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.0) -> pd.DataFrame:
        """
        Гибридный поиск с поддержкой количества и минимальной схожести.
        
        Args:
            query: текстовый запрос
            top_k: максимальное количество возвращаемых результатов
            min_similarity: минимальный порог схожести (0.0-1.0)
                
        Returns:
            DataFrame с результатами, удовлетворяющими обоим условиям
        """
        query_processed = self.preprocessor.preprocess(query, for_search=True)
        query_embedding = self.model.encode([query_processed], convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, self.meme_embeddings)[0]

        if min_similarity > 0:
            mask = similarities >= min_similarity
            eligible_indices = np.where(mask)[0]
        else:
            eligible_indices = np.arange(len(similarities))
        
        if len(eligible_indices) == 0:
            return pd.DataFrame()

        sorted_indices = eligible_indices[np.argsort(similarities[eligible_indices])[::-1]]
        k = min(int(top_k), len(sorted_indices))
        top_indices = sorted_indices[:k]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx].copy()
            row['score'] = float(similarities[idx])
            results.append(row)
        
        return pd.DataFrame(results)
    
    def get_image_bytes(self, idx: int) -> bytes:
        """Получить байты изображения по индексу в DataFrame."""
        return self.df.iloc[idx]['local_path']


st.set_page_config(
    page_title="🔍 Поиск мемов по смыслу",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_engine():
    """Загрузка поискового движка с кэшированием"""
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
    """Определение CSS-класса для бейджа в зависимости от оценки"""
    if score >= 0.7:
        return "high-score"
    elif score >= 0.4:
        return "medium-score"
    else:
        return "low-score"

def main():
    load_css()
    
    st.markdown('<h1 class="main-header">🔍 Поиск мемов по смыслу</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Опишите ситуацию на русском — найдем подходящий мем</p>', unsafe_allow_html=True)
    
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
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Введите запрос:",
            placeholder="Например: 'кот, который планирует месть'",
            key='query_input'
        )
    with col2:
        st.write("")
        st.write("")
        search_clicked = st.button("Искать", use_container_width=True, type="primary")
    
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
