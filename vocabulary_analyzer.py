from kiwipiepy import Kiwi
from typing import List, Dict, Set
from collections import Counter
import re

class VocabularyAnalyzer:
    def __init__(self):
        self.kiwi = Kiwi()
        
        # 어려운 단어 판단 기준
        # 금융 전문 용어
        self.financial_terms = {
            '압류', '가압류', '질권설정', '출연', '원가일', '결산일', '예금자보호법',
            '휴면예금', '거래중지계좌', '피해의심거래계좌', '사기이용계좌', '위법계약해지권',
            '금융소비자보호법', '민사집행법', '압류금지채권', '자료열람요구권',
            '저축예금', '입출식', '우대금리', '기본이자율', '세전', '결제실적',
            '자동이체', '연금수급', '공적연금', '휴면예금통합조회서비스'
        }
        
        # 법률 용어
        self.legal_terms = {
            '채무자', '채권자', '변제', '이행', '설정자', '담보물', '청구', '우선변제',
            '권리구제', '분쟁조정', '내부통제기준', '영업비밀', '침해'
        }
        
        # 한자어 기반 어려운 일반 용어
        self.difficult_hanja = {
            '충족', '해당', '제한', '적용', '산출', '지급', '해지', '양도', '설정',
            '충당', '확보', '임시', '부담', '증가', '경과', '완전', '일컬', '무상'
        }
        
        # 복합 표현
        self.complex_expressions = {
            '에 따라', '에 의해', '에 대하여', '에 관한', '을 위하여', '로써', '로서',
            '에 해당하는', '에 따른', '을 통한', '에 대한'
        }
        
    def calculate_vocabulary_diversity(self, tokens: List[str]) -> float:
        """어휘 다양성 비율 계산 (Type-Token Ratio)"""
        if not tokens:
            return 0
        
        unique_tokens = set(tokens)
        return len(unique_tokens) / len(tokens)
    
    def identify_difficult_words(self, sentence: str) -> Dict:
        """어려운 단어 식별 및 분류"""
        difficult_words = {
            'financial': [],
            'legal': [],
            'hanja': [],
            'complex': []
        }
        
        # 형태소 분석
        result = self.kiwi.tokenize(sentence)
        tokens = [token.form for token in result if token.tag.startswith('N')]
        
        for token in tokens:
            if token in self.financial_terms:
                difficult_words['financial'].append(token)
            elif token in self.legal_terms:
                difficult_words['legal'].append(token)
            elif token in self.difficult_hanja:
                difficult_words['hanja'].append(token)
        
        # 복합 표현 찾기
        for expr in self.complex_expressions:
            if expr in sentence:
                difficult_words['complex'].append(expr)
        
        return difficult_words
    
    def calculate_difficult_word_ratio(self, sentence: str) -> float:
        """어려운 단어 비율 계산"""
        result = self.kiwi.tokenize(sentence)
        tokens = [token.form for token in result]
        if not tokens:
            return 0
        
        difficult_words = self.identify_difficult_words(sentence)
        total_difficult = sum(len(words) for words in difficult_words.values())
        
        return total_difficult / len(tokens)
    
    def detect_sino_korean_ratio(self, sentence: str) -> float:
        """한자어 비율 계산"""
        # 한자어 패턴 (2글자 이상의 한자어)
        sino_pattern = re.compile(r'[가-힣]{2,}')
        
        result = self.kiwi.tokenize(sentence)
        tokens = [token.form for token in result if token.tag.startswith('N')]
        if not tokens:
            return 0
        
        # 간단한 휴리스틱: 2글자 이상 명사 중 받침이 특정 패턴인 경우
        sino_count = 0
        for token in tokens:
            if len(token) >= 2:
                # 한자어 추정 (마지막 글자가 특정 받침으로 끝나는 경우)
                if token[-1] in '력적성화용제도부처산출입금액':
                    sino_count += 1
        
        return sino_count / len(tokens) if tokens else 0
    
    def analyze_vocabulary_difficulty(self, sentence: str) -> Dict:
        """어휘 난이도 종합 분석"""
        # 형태소 분석
        result = self.kiwi.tokenize(sentence)
        pos_tags = [(token.form, token.tag) for token in result]
        all_tokens = [word for word, pos in pos_tags]
        nouns = [word for word, pos in pos_tags if pos.startswith('N')]
        
        # 어휘 지표 계산
        vocabulary_diversity = self.calculate_vocabulary_diversity(all_tokens)
        difficult_words = self.identify_difficult_words(sentence)
        difficult_word_ratio = self.calculate_difficult_word_ratio(sentence)
        sino_korean_ratio = self.detect_sino_korean_ratio(sentence)
        
        # 어휘 난이도 점수 계산 (0-10 스케일)
        vocab_difficulty = 0
        
        # 어휘 다양성 (다양할수록 어려움, 0-2점)
        if vocabulary_diversity > 0.8:
            vocab_difficulty += 2
        elif vocabulary_diversity > 0.6:
            vocab_difficulty += 1
        
        # 어려운 단어 비율 (0-4점)
        if difficult_word_ratio > 0.3:
            vocab_difficulty += 4
        elif difficult_word_ratio > 0.2:
            vocab_difficulty += 3
        elif difficult_word_ratio > 0.1:
            vocab_difficulty += 2
        elif difficult_word_ratio > 0:
            vocab_difficulty += 1
        
        # 한자어 비율 (0-2점)
        if sino_korean_ratio > 0.5:
            vocab_difficulty += 2
        elif sino_korean_ratio > 0.3:
            vocab_difficulty += 1
        
        # 전문용어 가중치 (0-2점)
        if difficult_words['financial'] or difficult_words['legal']:
            vocab_difficulty += 2
        
        # 최종 점수 정규화
        vocab_difficulty = min(vocab_difficulty, 10)
        
        return {
            'vocabulary_diversity': round(vocabulary_diversity, 3),
            'difficult_word_ratio': round(difficult_word_ratio, 3),
            'sino_korean_ratio': round(sino_korean_ratio, 3),
            'difficult_words': difficult_words,
            'total_difficult_words': sum(len(words) for words in difficult_words.values()),
            'vocab_difficulty_score': round(vocab_difficulty, 2)
        }