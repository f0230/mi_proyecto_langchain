import re
import string
from typing import List, Optional, Set
import unicodedata
from loguru import logger

try:
    from cleantext import clean
    CLEANTEXT_AVAILABLE = True
except ImportError:
    CLEANTEXT_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class TextCleaner:
    """Clase para limpiar y normalizar texto."""
    
    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        """Elimina espacios en blanco excesivos."""
        return re.sub(r'\s+', ' ', text).strip()
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """Elimina URLs del texto."""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', text)
    
    @staticmethod
    def remove_html_tags(text: str) -> str:
        """Elimina etiquetas HTML."""
        html_pattern = re.compile(r'<.*?>')
        return html_pattern.sub('', text)
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normaliza caracteres Unicode."""
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    @staticmethod
    def remove_special_chars(text: str, keep_punct: bool = True) -> str:
        """
        Elimina caracteres especiales.
        
        Args:
            text: Texto a limpiar
            keep_punct: Si es False, también elimina signos de puntuación
        """
        if keep_punct:
            # Mantener puntuación básica pero eliminar otros caracteres especiales
            pattern = re.compile(r'[^\w\s.,;:!?()\[\]{}\'"«»""''-]')
            return pattern.sub('', text)
        else:
            # Eliminar todos los caracteres que no sean alfanuméricos o espacios
            pattern = re.compile(r'[^\w\s]')
            return pattern.sub('', text)
    
    @staticmethod
    def remove_stopwords(text: str, language: str = 'spanish') -> str:
        """
        Elimina palabras vacías (stopwords).
        
        Args:
            text: Texto a limpiar
            language: Idioma para las stopwords
        """
        if not NLTK_AVAILABLE:
            logger.warning("NLTK no está disponible. No se eliminarán stopwords.")
            return text
        
        try:
            stop_words = set(stopwords.words(language))
            word_list = text.split()
            filtered_words = [word for word in word_list if word.lower() not in stop_words]
            return ' '.join(filtered_words)
        except Exception as e:
            logger.error(f"Error al eliminar stopwords: {str(e)}")
            return text
    
    @staticmethod
    def clean_text(
        text: str,
        remove_urls: bool = True,
        remove_html: bool = True,
        normalize_unicode: bool = True,
        remove_special: bool = False,
        keep_punct: bool = True,
        remove_extra_spaces: bool = True,
        remove_stops: bool = False,
        language: str = 'spanish'
    ) -> str:
        """
        Limpia el texto aplicando varios métodos según los parámetros.
        
        Args:
            text: Texto a limpiar
            remove_urls: Si es True, elimina URLs
            remove_html: Si es True, elimina etiquetas HTML
            normalize_unicode: Si es True, normaliza caracteres Unicode
            remove_special: Si es True, elimina caracteres especiales
            keep_punct: Si es True y remove_special es True, mantiene signos de puntuación
            remove_extra_spaces: Si es True, elimina espacios en blanco excesivos
            remove_stops: Si es True, elimina stopwords
            language: Idioma para las stopwords
            
        Returns:
            Texto limpio
        """
        if not text:
            return ""
        
        # Si clean-text está disponible, usamos su implementación optimizada
        if CLEANTEXT_AVAILABLE:
            try:
                cleaned = clean(
                    text,
                    fix_unicode=normalize_unicode,
                    to_ascii=normalize_unicode,
                    lower=False,
                    no_urls=remove_urls,
                    no_emails=True,
                    no_digits=False,
                    no_punct=remove_special and not keep_punct
                )
                if remove_html:
                    cleaned = TextCleaner.remove_html_tags(cleaned)
                if remove_extra_spaces:
                    cleaned = TextCleaner.remove_extra_whitespace(cleaned)
                if remove_stops:
                    cleaned = TextCleaner.remove_stopwords(cleaned, language)
                return cleaned
            except Exception as e:
                logger.warning(f"Error usando clean-text: {str(e)}. Fallback a implementación estándar.")
        
        # Implementación estándar
        if not isinstance(text, str):
            text = str(text)
        
        if remove_urls:
            text = TextCleaner.remove_urls(text)
        
        if remove_html:
            text = TextCleaner.remove_html_tags(text)
        
        if normalize_unicode:
            text = TextCleaner.normalize_unicode(text)
        
        if remove_special:
            text = TextCleaner.remove_special_chars(text, keep_punct)
        
        if remove_extra_spaces:
            text = TextCleaner.remove_extra_whitespace(text)
        
        if remove_stops:
            text = TextCleaner.remove_stopwords(text, language)
        
        return text