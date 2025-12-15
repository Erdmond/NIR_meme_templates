import re
import pymorphy3
import unicodedata

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
